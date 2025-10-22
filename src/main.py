import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main orchestrator: spawns `src.train` as a subprocess for a single run."""
    # ---------------------------------------------------------------------
    # Persist resolved config in results directory for bookkeeping
    # ---------------------------------------------------------------------
    results_dir = Path(cfg.results_dir).absolute()
    results_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(results_dir / "config.yaml"))

    # ---------------------------------------------------------------------
    # Compose Hydra override list to forward to child process (src.train)
    # ---------------------------------------------------------------------
    overrides: List[str] = [f"run={cfg.run.run_id}", f"results_dir={cfg.results_dir}"]

    if cfg.get("trial_mode", False):
        overrides.extend(
            [
                "trial_mode=true",
                "wandb.mode=disabled",
                "run.training.epochs=1",
                "run.optuna.n_trials=0",
            ]
        )

    # Ensure Hydra output directory inside the dedicated run folder (avoid clutter)
    overrides.append(f"hydra.run.dir={cfg.results_dir}/{cfg.run.run_id}/hydra")

    cmd = [sys.executable, "-u", "-m", "src.train"] + overrides
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
