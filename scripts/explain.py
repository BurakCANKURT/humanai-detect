"""Asama 7 (XAI): SHAP tabanli global ve lokal aciklanabilirlik.

Girdi : outputs/models/<name>.pkl + data/processed/fused.parquet
Cikti :
  outputs/figures/shap/<name>_bar.png    — global bar grafigi
  outputs/figures/shap/<name>.png        — global beeswarm grafigi
  outputs/figures/shap/<name>_local.png  — ornek-duzeyinde waterfall (--sample-idx ile)
  outputs/reports/shap_summary.md        — onem tablosu (Markdown)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.explainability.shap_analysis import run_shap_global, run_shap_local


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="SHAP analizi icin .pkl model dosyasi")
    parser.add_argument("--input", default=None, help="fused.parquet yolu")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Lokal waterfall icin ornek indeksi (varsayilan: 0)")
    parser.add_argument("--max-display", type=int, default=20,
                        help="Gosterilecek max ozellik sayisi")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    fused_path = Path(args.input) if args.input else PROJECT_ROOT / paths_cfg["processed_dir"] / "fused.parquet"

    if not fused_path.exists():
        print(f"[explain] {fused_path} bulunamadi. Once build_features.py calistirin.")
        return

    model_path = Path(args.model)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_parquet(fused_path)
    feat_cols = [c for c in df.columns if c not in ("sample_id", "label")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    feature_names = feat_cols

    model_stem = model_path.stem
    shap_dir = PROJECT_ROOT / paths_cfg["figures_dir"] / "shap"

    # Global SHAP
    print(f"[explain] Global SHAP analizi basliyor ({len(X)} ornek)...")
    run_shap_global(
        model, X, feature_names,
        out_path=shap_dir / f"{model_stem}.png",
        max_display=args.max_display,
    )

    # Lokal SHAP
    idx = args.sample_idx
    if idx < len(X):
        print(f"[explain] Lokal SHAP (ornek #{idx})...")
        run_shap_local(
            model, X[idx:idx+1], feature_names,
            out_path=shap_dir / f"{model_stem}_local_{idx}.png",
            X_background=X,
        )

    # Markdown onem raporu (top-20 global SHAP ortalamasi)
    _write_shap_report(model, X, feature_names, model_stem, paths_cfg)


def _write_shap_report(model, X, feature_names, model_stem, paths_cfg):
    import shap
    from humanai_detect.explainability.shap_analysis import _get_explainer

    explainer = _get_explainer(model, X)
    shap_values = explainer(X)

    # Mutlak ortalama SHAP degerleri
    if hasattr(shap_values, "values"):
        vals = np.abs(shap_values.values)
        if vals.ndim == 3:
            vals = vals.mean(axis=2)  # cok sinifli: sinif ortalamasi
        mean_shap = vals.mean(axis=0)
    else:
        mean_shap = np.abs(np.array(shap_values)).mean(axis=0)

    ranked = sorted(zip(feature_names, mean_shap.tolist()), key=lambda x: -x[1])

    lines = ["# SHAP Global Onem Raporu\n", "| Ozellik | Ort. |SHAP| |", "|---------|------------|"]
    for name, score in ranked[:20]:
        lines.append(f"| {name} | {score:.5f} |")

    report_path = PROJECT_ROOT / paths_cfg["reports_dir"] / "shap_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[explain] SHAP raporu -> {report_path}")


if __name__ == "__main__":
    main()
