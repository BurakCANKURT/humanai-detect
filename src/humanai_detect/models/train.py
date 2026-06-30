"""5-fold stratified CV ile model egitimi ve kayit."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from .factory import build_model


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def run_cv_training(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model_params: dict[str, Any],
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Stratified k-fold CV ile egitim yapar; fold metriklerini ve ozet istatistikleri dondurur.

    Donus sozlugu:
        fold_metrics : her fold icin accuracy/macro_f1/weighted_f1
        mean_*       : metrik ortalamasi
        std_*        : metrik standart sapmasi
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_results: list[dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        model = build_model(model_name, model_params)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        metrics = _fold_metrics(y[val_idx], y_pred)
        metrics["fold"] = float(fold)
        fold_results.append(metrics)
        print(
            f"  Fold {fold}/{cv_folds} | acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )

    summary: dict[str, Any] = {"fold_metrics": fold_results}
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        vals = [m[key] for m in fold_results]
        summary[f"mean_{key}"] = float(np.mean(vals))
        summary[f"std_{key}"] = float(np.std(vals))

    return summary


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model_params: dict[str, Any],
    save_path: Path | None = None,
) -> Any:
    """Tum egitim verisiyle final modeli egitir, opsiyonel olarak pickle ile kaydet.

    CV bittikten sonra en iyi hiperparametrelerle tam egitim icin cagrilir.
    """
    model = build_model(model_name, model_params)
    model.fit(X, y)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[train] Model kaydedildi -> {save_path}")

    return model
