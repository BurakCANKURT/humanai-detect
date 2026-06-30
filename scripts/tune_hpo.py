"""Asama 6 (HPO): Optuna ile hiperparametre optimizasyonu.

Girdi : data/processed/fused.parquet + configs/hpo_eval.yaml
Cikti :
  outputs/reports/hpo/<model>_best_params.yaml
  outputs/reports/hpo/<model>_study.json   (tum trial ozeti)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.models.hpo import run_optuna_study

LABEL_TO_INT = {"human": 0, "ai_raw": 1, "ai_humanized": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["xgboost", "catboost", "mlp", "logreg"],
        default="xgboost",
        help="HPO uygulanacak model",
    )
    parser.add_argument("--input", default=None, help="fused.parquet yolu")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Optuna trial sayisi (varsayilan: hpo_eval.yaml'dan)")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    hpo_cfg = load_yaml("hpo_eval")["hpo"]

    fused_path = Path(args.input) if args.input else PROJECT_ROOT / paths_cfg["processed_dir"] / "fused.parquet"
    if not fused_path.exists():
        print(f"[hpo] {fused_path} bulunamadi. Once build_features.py calistirin.")
        return

    df = pd.read_parquet(fused_path)
    feat_cols = [c for c in df.columns if c not in ("sample_id", "label")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].map(LABEL_TO_INT).to_numpy()

    n_trials = args.n_trials or hpo_cfg.get("n_trials", 50)
    metric = hpo_cfg.get("metric", "macro_f1")
    cv_folds = load_yaml("hpo_eval")["evaluation"]["cv_folds"]

    print(f"[hpo] {args.model} icin {n_trials} trial Optuna calisiyor (metrik={metric})...")
    study = run_optuna_study(
        args.model, X, y,
        n_trials=n_trials,
        metric=metric,
        cv_folds=cv_folds,
    )

    best = study.best_params
    best_value = study.best_value
    print(f"[hpo] En iyi {metric}: {best_value:.4f}")
    print(f"[hpo] En iyi parametreler: {best}")

    # Kaydet
    report_dir = PROJECT_ROOT / paths_cfg["reports_dir"] / "hpo"
    report_dir.mkdir(parents=True, exist_ok=True)

    params_path = report_dir / f"{args.model}_best_params.yaml"
    params_path.write_text(yaml.dump(best, allow_unicode=True), encoding="utf-8")
    print(f"[hpo] Best params -> {params_path}")

    # Trial ozeti
    trials_data = [
        {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
        for t in study.trials
    ]
    study_path = report_dir / f"{args.model}_study.json"
    study_path.write_text(json.dumps(trials_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hpo] Study JSON ({len(trials_data)} trial) -> {study_path}")


if __name__ == "__main__":
    main()
