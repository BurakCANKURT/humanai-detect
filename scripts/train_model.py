"""Asama 6: model egitimi (XGBoost/CatBoost/MLP/LogReg + stacking), 5-fold CV.

Girdi : data/processed/fused.parquet + configs/models.yaml
Cikti : outputs/models/<name>.pkl
        outputs/reports/cv_results/<name>_cv.json
        outputs/reports/cv_results/<name>_cv.md
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.models.train import run_cv_training, train_final_model
from humanai_detect.models.hierarchical import run_hierarchical_cv, train_final_hierarchical

LABEL_NAMES = ["human", "ai_raw", "ai_humanized"]
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(LABEL_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["xgboost", "catboost", "mlp", "logreg", "stacking", "hierarchical"],
        default="xgboost",
        help="Hangi model egitilecek",
    )
    parser.add_argument("--input", default=None, help="fused.parquet yolu")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-final", action="store_true",
                        help="Sadece CV yap, tam model kaydetme")
    parser.add_argument("--balance", action="store_true",
                        help="Her sinifi min sinif boyutuna subsample et")
    parser.add_argument("--no-group-cv", action="store_true",
                        help="StratifiedGroupKFold yerine eski (kaynak-korumasiz) StratifiedKFold kullan")
    parser.add_argument("--include-holdout", action="store_true",
                        help="holdout_ids.txt'deki orneklerin de CV/egitime dahil edilmesini sagla "
                             "(varsayilan: haric tutulur, bkz. make_holdout_split.py)")
    parser.add_argument("--tag", default=None,
                        help="Cikti dosya adina eklenecek ek etiket (orn. 'grouped')")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    models_cfg = load_yaml("models")
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]

    fused_path = Path(args.input) if args.input else processed_dir / "fused.parquet"
    if not fused_path.exists():
        print(f"[train] {fused_path} bulunamadi. Once build_features.py calistirin.")
        return

    df = pd.read_parquet(fused_path)

    groups_path = processed_dir / "groups.parquet"
    use_groups = groups_path.exists() and not args.no_group_cv
    if use_groups:
        groups_df = pd.read_parquet(groups_path)[["sample_id", "group_id"]]
        df = df.merge(groups_df, on="sample_id", how="left")
        if df["group_id"].isna().any():
            print("[train] UYARI: bazi orneklerde group_id yok, StratifiedKFold'a donuluyor.")
            use_groups = False

    holdout_path = processed_dir / "holdout_ids.txt"
    if holdout_path.exists() and not args.include_holdout:
        holdout_ids = set(holdout_path.read_text(encoding="utf-8").splitlines())
        n_before = len(df)
        df = df[~df["sample_id"].isin(holdout_ids)].reset_index(drop=True)
        print(f"[train] held-out set haric tutuldu: {n_before} -> {len(df)} ornek "
              f"({len(holdout_ids)} held-out)")

    if args.balance:
        min_n = df["label"].value_counts().min()
        df = pd.concat(
            [g.sample(n=min_n, random_state=42) for _, g in df.groupby("label")],
            ignore_index=True,
        )
        print(f"[train] Dengelendi: her siniftan {min_n} ornek (toplam {len(df)})")

    groups = df["group_id"].to_numpy() if use_groups else None

    feat_cols = [c for c in df.columns if c not in ("sample_id", "label", "group_id")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].map(LABEL_TO_INT).to_numpy()
    cv_kind = "StratifiedGroupKFold" if use_groups else "StratifiedKFold"
    print(f"[train] {len(X)} ornek, {len(feat_cols)} ozellik, model={args.model}, cv={cv_kind}")

    model_name = args.model
    common = models_cfg.get("common", {})
    if model_name == "stacking":
        model_params = models_cfg
    elif model_name == "hierarchical":
        hier_cfg = models_cfg.get("hierarchical", {})
        stage_a_name = hier_cfg.get("stage_a_model", "xgboost")
        stage_b_name = hier_cfg.get("stage_b_model", "xgboost")
        stage_a_params = {**common, **models_cfg.get(stage_a_name, {})}
        stage_b_params = {**common, **models_cfg.get(stage_b_name, {})}
    else:
        model_params = {**common, **models_cfg.get(model_name, {})}

    # CV
    print(f"[train] {args.cv_folds}-fold CV basliyor...")
    if model_name == "hierarchical":
        cv_results = run_hierarchical_cv(
            X, y, stage_a_name, stage_a_params, stage_b_name, stage_b_params,
            cv_folds=args.cv_folds, groups=groups,
        )
    else:
        cv_results = run_cv_training(
            X, y, model_name, model_params, cv_folds=args.cv_folds, groups=groups,
        )

    mean_f1 = cv_results["mean_macro_f1"]
    std_f1 = cv_results["std_macro_f1"]
    print(f"[train] CV macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")

    # CV raporunu kaydet
    name_tag = f"{model_name}_{args.tag}" if args.tag else model_name
    report_dir = PROJECT_ROOT / paths_cfg["reports_dir"] / "cv_results"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / f"{name_tag}_cv.json"
    json_path.write_text(json.dumps(cv_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] CV JSON -> {json_path}")

    md_lines = [
        f"# CV Sonuclari — {name_tag}",
        f"\n**Macro-F1:** {mean_f1:.4f} ± {std_f1:.4f}",
        f"**Accuracy:** {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}",
        "\n| Fold | macro_f1 | accuracy |",
        "|------|----------|----------|",
    ]
    for fold in cv_results.get("fold_metrics", []):
        md_lines.append(f"| {int(fold['fold'])} | {fold['macro_f1']:.4f} | {fold['accuracy']:.4f} |")
    (report_dir / f"{name_tag}_cv.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Tam model (tüm veri)
    if not args.no_final:
        models_dir = PROJECT_ROOT / paths_cfg["models_dir"]
        models_dir.mkdir(parents=True, exist_ok=True)
        save_path = models_dir / f"{name_tag}.pkl"
        if model_name == "hierarchical":
            model = train_final_hierarchical(
                X, y, stage_a_name, stage_a_params, stage_b_name, stage_b_params,
            )
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            print(f"[train] Model kaydedildi -> {save_path}")
        else:
            train_final_model(X, y, model_name, model_params, save_path=save_path)
            print(f"[train] Model kaydedildi -> {save_path}")


if __name__ == "__main__":
    main()
