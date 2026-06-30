"""SHAP tabanli model aciklanabilirlik analizi (global + lokal)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _get_explainer(model: Any, X_background: np.ndarray):
    """Model tipine gore uygun SHAP explainer'i olusturur."""
    import shap

    model_type = type(model).__name__.lower()
    if any(k in model_type for k in ("xgb", "catboost", "lgbm", "randomforest", "gradientboost")):
        return shap.TreeExplainer(model)
    if any(k in model_type for k in ("logistic", "linearsvc", "ridge")):
        try:
            return shap.LinearExplainer(model, X_background)
        except Exception:
            pass
    # Genel amaçli (yavash ama evrensel) — kucuk arka plan ornekleme ile
    background = shap.sample(X_background, min(50, len(X_background)))
    return shap.KernelExplainer(model.predict_proba, background)


def run_shap_global(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    max_display: int = 20,
) -> None:
    """Global SHAP onem dagilimi (bar + beeswarm) grafigini PNG olarak kaydeder.

    Yuksek onem -> o ozelligin sinif kararlarinda buyuk etkisi var.
    """
    import matplotlib.pyplot as plt
    import shap

    explainer = _get_explainer(model, X)
    shap_values = explainer(X)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Bar grafiği (ortalama mutlak SHAP degerleri)
    bar_path = Path(out_path).with_stem(Path(out_path).stem + "_bar")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] Global bar grafigi -> {bar_path}")

    # Beeswarm (dagilim)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] Global beeswarm grafigi -> {out_path}")


def run_shap_local(
    model: Any,
    sample: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    X_background: np.ndarray | None = None,
) -> None:
    """Tek bir ornek icin SHAP waterfall grafigini PNG olarak kaydeder.

    sample       : [1, n_features] veya [n_features] seklinde tek ornek
    X_background : KernelExplainer icin gerekli arka plan veriseti
    """
    import matplotlib.pyplot as plt
    import shap

    sample = np.atleast_2d(sample)
    bg = X_background if X_background is not None else sample
    explainer = _get_explainer(model, bg)
    shap_values = explainer(sample)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sv0 = shap_values[0]
    # Cok sinifli Explanation: values shape (n_features, n_classes) -> sinif 0
    if hasattr(sv0, "values") and sv0.values.ndim == 2:
        sv = shap.Explanation(
            values=sv0.values[:, 0],
            base_values=float(np.atleast_1d(sv0.base_values)[0]),
            data=sv0.data,
            feature_names=sv0.feature_names,
        )
    else:
        sv = sv0
    shap.plots.waterfall(sv, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] Lokal waterfall grafigi -> {out_path}")
