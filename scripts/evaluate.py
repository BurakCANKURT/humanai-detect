"""Asama 7: degerlendirme (confusion matrix, F1, macro-F1, ROC-AUC, UMAP).

Girdi : outputs/models/<name>.pkl + data/processed/fused.parquet
Cikti :
  outputs/reports/cv_results/<name>.md
  outputs/figures/confusion/<name>.png
  outputs/figures/roc/<name>.png
  outputs/figures/umap/<name>.png
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.evaluation.metrics import compute_metrics, format_metrics_report
from humanai_detect.evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_umap_projection,
)

LABEL_NAMES = ["human", "ai_raw", "ai_humanized"]
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(LABEL_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Degerlendirme icin .pkl model dosyasi")
    parser.add_argument("--input", default=None, help="fused.parquet yolu")
    parser.add_argument("--split", choices=["all", "test"], default="all",
                        help="Hangi split uzerinde degerlendirilsin (simdilik 'all')")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    fused_path = Path(args.input) if args.input else PROJECT_ROOT / paths_cfg["processed_dir"] / "fused.parquet"

    if not fused_path.exists():
        print(f"[evaluate] {fused_path} bulunamadi. Once build_features.py calistirin.")
        return

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[evaluate] {model_path} bulunamadi. Once train_model.py calistirin.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_parquet(fused_path)
    feat_cols = [c for c in df.columns if c not in ("sample_id", "label")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].map(LABEL_TO_INT).to_numpy()

    y_pred = np.array(model.predict(X)).reshape(-1)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    metrics = compute_metrics(y, y_pred, y_proba=y_proba, label_names=LABEL_NAMES)

    # --- Rapor ---
    model_stem = model_path.stem
    report_dir = PROJECT_ROOT / paths_cfg["reports_dir"] / "cv_results"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{model_stem}.md"
    report_path.write_text(format_metrics_report(metrics), encoding="utf-8")
    print(f"[evaluate] Rapor -> {report_path}")
    print(f"  accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")

    # --- Gorseller ---
    fig_dir = PROJECT_ROOT / paths_cfg["figures_dir"]

    plot_confusion_matrix(
        y, y_pred, LABEL_NAMES,
        out_path=fig_dir / "confusion" / f"{model_stem}.png",
        title=f"Confusion Matrix — {model_stem}",
    )

    if y_proba is not None:
        plot_roc_curves(
            y, y_proba, LABEL_NAMES,
            out_path=fig_dir / "roc" / f"{model_stem}.png",
            title=f"ROC Curves — {model_stem}",
        )

    # UMAP: embed dogruca X uzerinde calis (egerl buyuk ise alt-ornekle)
    n_umap = min(500, len(X))
    idx = np.random.default_rng(42).choice(len(X), n_umap, replace=False)
    plot_umap_projection(
        X[idx], y[idx],
        out_path=fig_dir / "umap" / f"{model_stem}.png",
        label_names=LABEL_NAMES,
        title=f"UMAP — {model_stem}",
    )


if __name__ == "__main__":
    main()
