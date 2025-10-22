import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy.stats import ttest_ind

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _learning_curve(df: pd.DataFrame, run_id: str, out_dir: Path) -> Path:
    plt.figure(figsize=(8, 5))
    if "val_acc" in df.columns:
        plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    if "train_acc" in df.columns:
        plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve – {run_id}")
    plt.legend()
    plt.tight_layout()
    file_path = out_dir / f"{run_id}_learning_curve.pdf"
    plt.savefig(file_path)
    plt.close()
    return file_path


def _confusion_matrix(cm: List[List[int]], run_id: str, out_dir: Path) -> Path:
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix – {run_id}")
    plt.tight_layout()
    file_path = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(file_path)
    plt.close()
    return file_path

# -----------------------------------------------------------------------------
# Evaluation script (stand-alone)
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help="JSON string list of run IDs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).absolute()
    run_ids: List[str] = json.loads(args.run_ids)

    cfg = OmegaConf.load(results_dir / "config.yaml")
    entity, project = cfg.wandb.entity, cfg.wandb.project

    api = wandb.Api()
    aggregated: Dict[str, Dict[str, Any]] = {}
    generated_files: List[Path] = []

    # ------------------- per-run processing ----------------------------------
    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        hist_df = run.history(keys=None)  # full metric history
        summary = run.summary._json_dict
        run_cfg = dict(run.config)

        run_dir = results_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save full history
        metrics_path = run_dir / "metrics.json"
        _save_json(hist_df.to_dict(orient="list"), metrics_path)
        generated_files.append(metrics_path)

        # Learning curve figure
        lc_path = _learning_curve(hist_df, rid, run_dir)
        generated_files.append(lc_path)

        # Confusion matrix, if present
        if "confusion_matrix" in summary and summary["confusion_matrix"] is not None:
            cm_path = _confusion_matrix(summary["confusion_matrix"], rid, run_dir)
            generated_files.append(cm_path)

        aggregated[rid] = {
            "best_val_acc": summary.get("best_val_acc", np.nan),
            "training_time": summary.get("best_training_time", np.nan),
            "method": run_cfg.get("run", {}).get("method", run_cfg.get("method", "unknown")),
        }

    # ------------------- aggregated analysis ---------------------------------
    cmp_dir = results_dir / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    agg_metrics_path = cmp_dir / "aggregated_metrics.json"
    _save_json(aggregated, agg_metrics_path)
    generated_files.append(agg_metrics_path)

    df = pd.DataFrame.from_dict(aggregated, orient="index")

    # Improvement vs baseline (first alphabetical method containing 'baseline' or 'comparative')
    methods = df["method"].unique()
    baseline_method = next((m for m in methods if "baseline" in str(m).lower() or "comparative" in str(m).lower()), None)
    improvement = {}
    if baseline_method is not None:
        base_vals = df[df["method"] == baseline_method]["best_val_acc"].astype(float).values
        for m in methods:
            if m == baseline_method:
                continue
            oth_vals = df[df["method"] == m]["best_val_acc"].astype(float).values
            if len(base_vals) and len(oth_vals):
                imp = (oth_vals.mean() - base_vals.mean()) / base_vals.mean()
                stat, p_val = ttest_ind(oth_vals, base_vals, equal_var=False)
                improvement[m] = {"improvement_vs_baseline": imp, "p_value": p_val}

    derived_path = cmp_dir / "derived_metrics.json"
    _save_json(improvement, derived_path)
    generated_files.append(derived_path)

    # ------------------- figures --------------------------------------------
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df.index, y="best_val_acc", hue="method", data=df)
    for idx, val in enumerate(df["best_val_acc"].astype(float)):
        plt.text(idx, val + 0.001, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("Best Val Accuracy")
    plt.title("Best Validation Accuracy per Run")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    bar_path = cmp_dir / "comparison_best_val_acc_bar_chart.pdf"
    plt.savefig(bar_path)
    plt.close()
    generated_files.append(bar_path)

    # Print all generated file paths (for GitHub Actions log parsing)
    for p in generated_files:
        print(p)


if __name__ == "__main__":
    main()
