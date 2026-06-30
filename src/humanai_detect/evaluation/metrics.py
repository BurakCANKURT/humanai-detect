"""Siniflandirma degerlendirme metrikleri: accuracy, F1, ROC-AUC, confusion matrix."""

from __future__ import annotations

from typing import Any

import numpy as np


LABEL_NAMES = ["human", "ai_raw", "ai_humanized"]


def compute_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """accuracy, macro_f1, per_class_f1, roc_auc_ovr ve confusion_matrix iceren sozluk dondurur.

    y_true / y_pred : tamsayi veya string dizileri
    y_proba         : [n_samples, n_classes] olasilik matrisi (ROC-AUC icin gerekli)
    label_names     : sinif isim listesi (varsayilan: LABEL_NAMES)
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = label_names or LABEL_NAMES

    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {
            lbl: float(s)
            for lbl, s in zip(
                labels,
                f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(labels)))),
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "label_names": labels,
    }

    if y_proba is not None:
        try:
            result["roc_auc_ovr"] = float(
                roc_auc_score(y_true, np.asarray(y_proba), multi_class="ovr", average="macro")
            )
        except Exception:
            result["roc_auc_ovr"] = float("nan")
    else:
        result["roc_auc_ovr"] = float("nan")

    return result


def format_metrics_report(metrics: dict[str, Any]) -> str:
    """compute_metrics ciktisini Markdown tablosu olarak bicimlendirip dondurur."""
    lines = [
        "## Degerlendirme Sonuclari\n",
        f"| Metrik | Deger |",
        f"|--------|-------|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Macro-F1 | {metrics['macro_f1']:.4f} |",
        f"| Weighted-F1 | {metrics['weighted_f1']:.4f} |",
        f"| ROC-AUC (OvR) | {metrics['roc_auc_ovr']:.4f} |",
        "",
        "### Sinif Bazli F1",
        "| Sinif | F1 |",
        "|-------|----|",
    ]
    for lbl, score in metrics.get("per_class_f1", {}).items():
        lines.append(f"| {lbl} | {score:.4f} |")
    return "\n".join(lines)
