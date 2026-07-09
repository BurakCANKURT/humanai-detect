"""Iki asamali (cascade) siniflandirma: once human/AI, sonra ai_raw/ai_humanized.

Motivasyon: confusion matrix'te human sinifi neredeyse kusursuz ayriliyor (F1~0.996),
asil zorluk ai_raw<->ai_humanized ayrimi. Bu modul, modelin kapasitesini dogrudan
zor olan ikinci ayrima yonlendirir.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .factory import build_model

HUMAN, AI_RAW, AI_HUMANIZED = 0, 1, 2


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def _fit_predict_fold(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
    stage_a_model: str, stage_a_params: dict[str, Any],
    stage_b_model: str, stage_b_params: dict[str, Any],
) -> np.ndarray:
    """Bir fold icin iki asamali modeli egitir, val seti icin 3-sinifli tahmin dondurur."""
    y_a_train = (y_train != HUMAN).astype(int)  # 0=human, 1=AI (raw ya da humanized)
    model_a = build_model(stage_a_model, stage_a_params)
    model_a.fit(X_train, y_a_train)

    ai_mask_train = y_train != HUMAN
    y_b_train = (y_train[ai_mask_train] == AI_HUMANIZED).astype(int)  # 0=ai_raw, 1=ai_humanized
    model_b = build_model(stage_b_model, stage_b_params)
    model_b.fit(X_train[ai_mask_train], y_b_train)

    y_a_pred = model_a.predict(X_val)
    y_pred = np.full(len(X_val), HUMAN, dtype=int)
    ai_val_idx = np.flatnonzero(y_a_pred == 1)
    if len(ai_val_idx) > 0:
        y_b_pred = model_b.predict(X_val[ai_val_idx])
        y_pred[ai_val_idx] = np.where(y_b_pred == 1, AI_HUMANIZED, AI_RAW)

    return y_pred


def run_hierarchical_cv(
    X: np.ndarray,
    y: np.ndarray,
    stage_a_model: str,
    stage_a_params: dict[str, Any],
    stage_b_model: str,
    stage_b_params: dict[str, Any],
    cv_folds: int = 5,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Iki asamali cascade icin stratified k-fold CV.

    groups verilirse StratifiedGroupKFold kullanilir (kaynak-duzeyinde leakage onleme).

    Donus, run_cv_training ile ayni sekilde: fold_metrics + mean_*/std_* (3-sinifli
    accuracy/macro_f1/weighted_f1 uzerinden, tek-asamali modellerle dogrudan kiyaslanabilir).
    """
    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        split_iter = skf.split(X, y, groups=groups)
    else:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        split_iter = skf.split(X, y)
    fold_results: list[dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(split_iter, 1):
        y_pred = _fit_predict_fold(
            X[train_idx], y[train_idx], X[val_idx],
            stage_a_model, stage_a_params, stage_b_model, stage_b_params,
        )
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


class HierarchicalClassifier:
    """Egitilmis iki asamali modeli tek bir nesnede tasiyan, pickle'lanabilir sarmalayici."""

    def __init__(self, model_a: Any, model_b: Any) -> None:
        self.model_a = model_a
        self.model_b = model_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_a_pred = self.model_a.predict(X)
        y_pred = np.full(len(X), HUMAN, dtype=int)
        ai_idx = np.flatnonzero(y_a_pred == 1)
        if len(ai_idx) > 0:
            y_b_pred = self.model_b.predict(X[ai_idx])
            y_pred[ai_idx] = np.where(y_b_pred == 1, AI_HUMANIZED, AI_RAW)
        return y_pred


def train_final_hierarchical(
    X: np.ndarray,
    y: np.ndarray,
    stage_a_model: str,
    stage_a_params: dict[str, Any],
    stage_b_model: str,
    stage_b_params: dict[str, Any],
) -> HierarchicalClassifier:
    """Tum veriyle iki asamali final modeli egitir."""
    y_a = (y != HUMAN).astype(int)
    model_a = build_model(stage_a_model, stage_a_params)
    model_a.fit(X, y_a)

    ai_mask = y != HUMAN
    y_b = (y[ai_mask] == AI_HUMANIZED).astype(int)
    model_b = build_model(stage_b_model, stage_b_params)
    model_b.fit(X[ai_mask], y_b)

    return HierarchicalClassifier(model_a, model_b)
