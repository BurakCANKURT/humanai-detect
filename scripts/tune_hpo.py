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
    parser.add_argument("--balance", action="store_true",
                        help="Her sinifi min sinif boyutuna subsample et (train_model.py --balance ile ayni havuz)")
    parser.add_argument("--no-group-cv", action="store_true",
                        help="StratifiedGroupKFold yerine eski (kaynak-korumasiz) StratifiedKFold kullan")
    parser.add_argument("--include-holdout", action="store_true",
                        help="holdout_ids.txt'deki orneklerin de HPO'ya dahil edilmesini sagla (varsayilan: haric)")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    hpo_cfg = load_yaml("hpo_eval")["hpo"]
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]

    fused_path = Path(args.input) if args.input else processed_dir / "fused.parquet"
    if not fused_path.exists():
        print(f"[hpo] {fused_path} bulunamadi. Once build_features.py calistirin.")
        return

    df = pd.read_parquet(fused_path)

    groups_path = processed_dir / "groups.parquet"
    use_groups = groups_path.exists() and not args.no_group_cv
    if use_groups:
        groups_df = pd.read_parquet(groups_path)[["sample_id", "group_id"]]
        df = df.merge(groups_df, on="sample_id", how="left")
        if df["group_id"].isna().any():
            print("[hpo] UYARI: bazi orneklerde group_id yok, StratifiedKFold'a donuluyor.")
            use_groups = False

    holdout_path = processed_dir / "holdout_ids.txt"
    if holdout_path.exists() and not args.include_holdout:
        holdout_ids = set(holdout_path.read_text(encoding="utf-8").splitlines())
        n_before = len(df)
        df = df[~df["sample_id"].isin(holdout_ids)].reset_index(drop=True)
        print(f"[hpo] held-out set haric tutuldu: {n_before} -> {len(df)} ornek ({len(holdout_ids)} held-out)")

    if args.balance:
        min_n = df["label"].value_counts().min()
        df = pd.concat(
            [g.sample(n=min_n, random_state=42) for _, g in df.groupby("label")],
            ignore_index=True,
        )
        print(f"[hpo] Dengelendi: her siniftan {min_n} ornek (toplam {len(df)})")

    groups = df["group_id"].to_numpy() if use_groups else None
    feat_cols = [c for c in df.columns if c not in ("sample_id", "label", "group_id")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].map(LABEL_TO_INT).to_numpy()

    n_trials = args.n_trials or hpo_cfg.get("n_trials", 50)
    metric = hpo_cfg.get("metric", "macro_f1")
    cv_folds = load_yaml("hpo_eval")["evaluation"]["cv_folds"]

    cv_kind = "StratifiedGroupKFold" if use_groups else "StratifiedKFold"
    print(f"[hpo] {len(X)} ornek, cv={cv_kind}, {args.model} icin {n_trials} trial Optuna calisiyor (metrik={metric})...")
    result = run_optuna_study(
        args.model, X, y,
        n_trials=n_trials,
        metric=metric,
        cv_folds=cv_folds,
        groups=groups,
    )

    best = result["best_params"]
    best_value = result["best_value"]
    print(f"[hpo] En iyi {metric}: {best_value:.4f}")
    print(f"[hpo] En iyi parametreler: {best}")

    # Kaydet
    report_dir = PROJECT_ROOT / paths_cfg["reports_dir"] / "hpo"
    report_dir.mkdir(parents=True, exist_ok=True)

    params_path = report_dir / f"{args.model}_best_params.yaml"
    params_path.write_text(yaml.dump(best, allow_unicode=True), encoding="utf-8")
    print(f"[hpo] Best params -> {params_path}")

    # Trial ozeti
    trials_data = result["trials"]
    study_path = report_dir / f"{args.model}_study.json"
    study_path.write_text(json.dumps(trials_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hpo] Study JSON ({len(trials_data)} trial) -> {study_path}")


if __name__ == "__main__":
    main()
