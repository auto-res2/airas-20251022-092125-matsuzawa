import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from src.model import build_model
from src.preprocess import get_dataloaders

# -----------------------------------------------------------------------------
# Global cache & environment variables (forced to ./.cache)
# -----------------------------------------------------------------------------
CACHE_ROOT = Path(".cache").absolute()
CACHE_ROOT.mkdir(exist_ok=True)
os.environ.update(
    {
        "TORCH_HOME": str(CACHE_ROOT / "torch"),
        "HF_HOME": str(CACHE_ROOT / "hf"),
        "HF_DATASETS_CACHE": str(CACHE_ROOT / "hf" / "datasets"),
        "WANDB_CACHE_DIR": str(CACHE_ROOT / "wandb"),
    }
)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed all relevant RNGs for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _log_step(wb_run: Optional["wandb.sdk.wandb_run.Run"], payload: Dict[str, Any], step: int) -> None:
    """Log a payload dict to WandB if enabled."""
    if wb_run is not None:
        wb_run.log(payload, step=step)

# -----------------------------------------------------------------------------
# Single supervised training routine (internal use by BOIL/BOIL-UC and baseline)
# -----------------------------------------------------------------------------

def _train_once(run_cfg: DictConfig, wb_run: Optional["wandb.sdk.wandb_run.Run"], global_step_offset: int = 0) -> Dict[str, Any]:
    """Train for `run_cfg.training.epochs` (or 1 epoch in trial mode).

    Returns a dict with keys: val_acc, val_loss, training_time, confusion_matrix
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(run_cfg.get("seed", 42)))

    train_loader, val_loader, _ = get_dataloaders(run_cfg)

    model = build_model(run_cfg)
    model.to(device)

    # ----------------- optimiser -------------------------------------------
    opt_name = str(run_cfg.training.optimizer).lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(run_cfg.training.learning_rate),
            momentum=float(run_cfg.training.momentum),
            weight_decay=float(run_cfg.training.weight_decay),
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(run_cfg.training.learning_rate),
            weight_decay=float(run_cfg.training.weight_decay),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {run_cfg.training.optimizer}")

    # ----------------- LR scheduler ----------------------------------------
    sched_name = str(run_cfg.training.scheduler).lower()
    scheduler = None
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(run_cfg.training.epochs)
        )

    criterion = nn.CrossEntropyLoss()
    best_val_acc, best_cm = 0.0, None
    tic = time.time()

    for epoch in range(int(run_cfg.training.epochs)):
        # ---------------- training loop ------------------------------------
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for b_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

            _log_step(
                wb_run,
                {
                    "batch_train_loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=global_step_offset + epoch * len(train_loader) + b_idx,
            )

            # ---------------- trial-mode early exit ------------------------
            if run_cfg.get("trial_mode", False) and b_idx >= 1:
                break

        train_loss_epoch = running_loss / max(1, running_total)
        train_acc_epoch = running_correct / max(1, running_total)

        # ---------------- validation loop ----------------------------------
        model.eval()
        v_losses, v_preds, v_targets = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                v_losses.append(loss.item())
                v_preds.extend(logits.argmax(dim=1).cpu().numpy())
                v_targets.extend(labels.cpu().numpy())

        val_loss_epoch = float(np.mean(v_losses))
        val_acc_epoch = float(accuracy_score(v_targets, v_preds))
        cm_epoch = confusion_matrix(v_targets, v_preds).tolist()

        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            best_cm = cm_epoch

        _log_step(
            wb_run,
            {
                "epoch": epoch,
                "train_loss": train_loss_epoch,
                "train_acc": train_acc_epoch,
                "val_loss": val_loss_epoch,
                "val_acc": val_acc_epoch,
            },
            step=global_step_offset + epoch,
        )

        if run_cfg.get("trial_mode", False):
            break  # run only 1 epoch in trial mode

        if scheduler is not None:
            scheduler.step()

    training_time = time.time() - tic
    return {
        "val_acc": best_val_acc,
        "val_loss": val_loss_epoch,
        "training_time": training_time,
        "confusion_matrix": best_cm,
    }

# -----------------------------------------------------------------------------
# BOIL / BOIL-UC implementation (learning-rate optimisation only)
# -----------------------------------------------------------------------------

def _encode_lr(lr: float) -> np.ndarray:
    """Log-scale encode the learning rate as 1-D feature vector."""
    return np.array([[np.log10(lr)]], dtype=np.float64)  # shape (1,1)


def _random_lr(low: float = 1e-4, high: float = 5e-1) -> float:
    """Sample a LR uniformly in log-space between `low` and `high`."""
    return float(10 ** np.random.uniform(np.log10(low), np.log10(high)))


def _expected_improvement(mu: float, sigma: float, y_best: float, xi: float = 1e-2) -> float:
    """Compute EI for a normal distribution with mean `mu` and std `sigma`."""
    sigma = max(sigma, 1e-9)
    gamma = (mu - y_best - xi) / sigma
    return float(sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma)))


def _init_cost_model(cost_cfg: DictConfig):
    ctype = str(cost_cfg.type).lower()
    if ctype == "bayesianridge":
        return BayesianRidge(
            alpha_1=float(cost_cfg.alpha_1),
            lambda_1=float(cost_cfg.lambda_1),
            compute_score=False,
        )
    elif ctype == "linearregression":
        return LinearRegression()
    else:
        raise ValueError(f"Unsupported cost model: {cost_cfg.type}")


def run_boil(root_cfg: DictConfig, wb_run: Optional["wandb.sdk.wandb_run.Run"]) -> None:
    """Execute BOIL/BOIL-UC according to `root_cfg.run` settings."""
    run_cfg, algo_cfg, cost_cfg = root_cfg.run, root_cfg.run.algorithm, root_cfg.run.cost_model

    bo_iters = int(algo_cfg.bo_iterations)
    init_random = int(algo_cfg.get("init_random", 5))
    random_candidates = int(algo_cfg.get("random_candidates", 64))
    beta_uncert = float(algo_cfg.get("beta_uncert", 0.0))

    # Surrogate for utility (validation accuracy)
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
    cost_model = _init_cost_model(cost_cfg)

    X_features, y_util, y_cost = [], [], []  # design matrix & targets
    global_step_offset = 0

    for it in range(bo_iters):
        # ---------------- Candidate suggestion ---------------------------------
        if it < init_random or len(X_features) < 2:
            cand_lr = _random_lr()
        else:
            gp.fit(np.vstack(X_features), np.asarray(y_util))
            cost_model.fit(np.vstack(X_features), np.asarray(y_cost))

            candidate_lrs = [_random_lr() for _ in range(random_candidates)]
            acq_scores: List[float] = []
            for lr in candidate_lrs:
                feat = _encode_lr(lr)
                mu_u, std_u = gp.predict(feat, return_std=True)
                mu_u, std_u = float(mu_u.squeeze()), float(std_u.squeeze())
                ei_val = _expected_improvement(mu_u, std_u, y_best=max(y_util))
                if ei_val <= 0:
                    acq_scores.append(-np.inf)
                    continue

                if str(algo_cfg.name).lower() == "boil-uc":
                    # Uncertainty-aware denominator
                    if hasattr(cost_model, "predict") and "return_std" in cost_model.predict.__code__.co_varnames:
                        mu_c, std_c = cost_model.predict(feat, return_std=True)
                        mu_c, std_c = float(mu_c.squeeze()), float(std_c.squeeze())
                    else:
                        mu_c, std_c = float(cost_model.predict(feat)[0]), 0.0
                    denom = mu_c + beta_uncert * std_c + 1e-6
                else:  # plain BOIL
                    mu_c = float(cost_model.predict(feat)[0])
                    denom = mu_c + 1e-6

                acq_scores.append(np.log(ei_val) - np.log(denom))

            cand_lr = candidate_lrs[int(np.argmax(acq_scores))]

        # ---------------- Evaluate candidate -----------------------------------
        cand_cfg = OmegaConf.create(OmegaConf.to_container(run_cfg, resolve=True))
        cand_cfg.training.learning_rate = cand_lr
        cand_cfg.trial_mode = root_cfg.trial_mode  # propagate flag
        res = _train_once(cand_cfg, wb_run, global_step_offset)

        # ---------------- Update datasets --------------------------------------
        X_features.append(_encode_lr(cand_lr))
        y_util.append(float(res["val_acc"]))
        y_cost.append(float(res["training_time"]))

        if wb_run is not None:
            wb_run.log(
                {
                    "bo_iteration": it,
                    "candidate_lr": cand_lr,
                    "iter_val_acc": res["val_acc"],
                    "iter_training_time": res["training_time"],
                },
                step=it,
            )

        # large offset to separate batches between BO iterations in WandB UI
        global_step_offset += int(run_cfg.training.epochs) * 10_000

        if root_cfg.trial_mode:
            break  # single iteration in trial mode

    # ---------------- Summaries ------------------------------------------------
    if wb_run is not None and y_util:
        best_idx = int(np.argmax(y_util))
        wb_run.summary["best_val_acc"] = y_util[best_idx]
        wb_run.summary["best_learning_rate"] = 10 ** X_features[best_idx][0][0]
        wb_run.summary["best_training_time"] = y_cost[best_idx]

# -----------------------------------------------------------------------------
# Entry-point (hydra)
# -----------------------------------------------------------------------------


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Ensure results directory exists & save full resolved config for reproducibility
    results_dir = Path(cfg.results_dir).absolute()
    results_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(results_dir / "config.yaml"))

    # -------------- WandB initialisation --------------------------------------
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if cfg.trial_mode or cfg.wandb.mode == "disabled" or not wandb_api_key:
        os.environ["WANDB_MODE"] = "disabled"
        wb_run = None
    else:
        wb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(results_dir / cfg.run.run_id),
        )

    # ---------------- Dispatch algorithm --------------------------------------
    algo_name = str(cfg.run.algorithm.name).lower()
    if algo_name in {"boil", "boil-uc"}:
        run_boil(cfg, wb_run)
    else:  # plain single training run (sanity / ablation)
        run_cfg = deepcopy(cfg.run)
        run_cfg.trial_mode = cfg.trial_mode
        res = _train_once(run_cfg, wb_run)
        if wb_run is not None:
            wb_run.summary["best_val_acc"] = res["val_acc"]
            wb_run.summary["training_time"] = res["training_time"]
            wb_run.summary["confusion_matrix"] = res["confusion_matrix"]

    if wb_run is not None:
        print(f"WandB URL: {wb_run.url}")
        wb_run.finish()


if __name__ == "__main__":
    main()
