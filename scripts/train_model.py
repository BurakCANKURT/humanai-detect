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
from humanai_detect.models.stacking import build_stacking_from_config

LABEL_NAMES = ["human", "ai_raw", "ai_humanized"]
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(LABEL_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["xgboost", "catboost", "mlp", "logreg", "stacking"],
        default="xgboost",
        help="Hangi model egitilecek",
    )
    parser.add_argument("--input", default=None, help="fused.parquet yolu")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-final", action="store_true",
                        help="Sadece CV yap, tam model kaydetme")
    parser.add_argument("--balance", action="store_true",
                        help="Her sinifi min sinif boyutuna subsample et")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    models_cfg = load_yaml("models")

    fused_path = Path(args.input) if args.input else PROJECT_ROOT / paths_cfg["processed_dir"] / "fused.parquet"
    if not fused_path.exists():
        print(f"[train] {fused_path} bulunamadi. Once build_features.py calistirin.")
        return

    df = pd.read_parquet(fused_path)

    if args.balance:
        min_n = df["label"].value_counts().min()
        df = pd.concat(
            [g.sample(n=min_n, random_state=42) for _, g in df.groupby("label")],
            ignore_index=True,
        )
        print(f"[train] Dengelendi: her siniftan {min_n} ornek (toplam {len(df)})")

    feat_cols = [c for c in df.columns if c not in ("sample_id", "label")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].map(LABEL_TO_INT).to_numpy()
    print(f"[train] {len(X)} ornek, {len(feat_cols)} ozellik, model={args.model}")

    model_name = args.model
    if model_name == "stacking":
        model_params = {}
    else:
        common = models_cfg.get("common", {})
        model_params = {**common, **models_cfg.get(model_name, {})}

    # CV
    print(f"[train] {args.cv_folds}-fold CV basliyor...")
    cv_results = run_cv_training(X, y, model_name, model_params, cv_folds=args.cv_folds)

    mean_f1 = cv_results["mean_macro_f1"]
    std_f1 = cv_results["std_macro_f1"]
    print(f"[train] CV macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")

    # CV raporunu kaydet
    report_dir = PROJECT_ROOT / paths_cfg["reports_dir"] / "cv_results"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / f"{model_name}_cv.json"
    json_path.write_text(json.dumps(cv_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] CV JSON -> {json_path}")

    md_lines = [
        f"# CV Sonuclari — {model_name}",
        f"\n**Macro-F1:** {mean_f1:.4f} ± {std_f1:.4f}",
        f"**Accuracy:** {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}",
        "\n| Fold | macro_f1 | accuracy |",
        "|------|----------|----------|",
    ]
    for fold in cv_results.get("fold_metrics", []):
        md_lines.append(f"| {int(fold['fold'])} | {fold['macro_f1']:.4f} | {fold['accuracy']:.4f} |")
    (report_dir / f"{model_name}_cv.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Tam model (tüm veri)
    if not args.no_final:
        models_dir = PROJECT_ROOT / paths_cfg["models_dir"]
        models_dir.mkdir(parents=True, exist_ok=True)
        save_path = models_dir / f"{model_name}.pkl"

        if model_name == "stacking":
            model = build_stacking_from_config(models_cfg)
            model.fit(X, y)
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            print(f"[train] Stacking modeli kaydedildi -> {save_path}")
        else:
            train_final_model(X, y, model_name, model_params, save_path=save_path)
            print(f"[train] Model kaydedildi -> {save_path}")


if __name__ == "__main__":
    main()
