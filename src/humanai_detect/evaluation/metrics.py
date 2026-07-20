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


def compute_metrics_by_generator(
    y_true: Any,
    y_pred: Any,
    sample_ids: Any,
    y_proba: Any = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """sample_id'den cikarilan uretici (human/qwen/gpt4o_mini/claude_sonnet5) bazinda
    ayri compute_metrics ceagirir.

    Held-out sette tek bir toplam Accuracy/Macro-F1 raporlamak, model bu ureticileri
    ne kadar farkli ogrendigini gizler (bkz. proje notlarindaki Qwen-ai_raw 0.794 vs
    GPT-4o-mini-ai_raw 0.979 bulgusu) -- bu fonksiyon her `evaluate.py`/
    `measure_calibration.py` calistirmasinda bu kirilimi otomatik uretmek icin var,
    elle/ad-hoc hesaplamaya gerek kalmasin diye.
    """
    from humanai_detect.utils.generator_id import infer_generator

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba_arr = np.asarray(y_proba) if y_proba is not None else None
    generators = np.array([infer_generator(sid) for sid in sample_ids])

    result: dict[str, Any] = {}
    for gen in sorted(set(generators)):
        mask = generators == gen
        gen_metrics = compute_metrics(
            y_true[mask],
            y_pred[mask],
            y_proba=y_proba_arr[mask] if y_proba_arr is not None else None,
            label_names=label_names,
        )
        result[gen] = {
            "n": int(mask.sum()),
            "accuracy": gen_metrics["accuracy"],
            "macro_f1": gen_metrics["macro_f1"],
            "per_class_f1": gen_metrics["per_class_f1"],
        }
    return result


def format_generator_report(by_generator: dict[str, Any]) -> str:
    """compute_metrics_by_generator ciktisini Markdown tablosu olarak bicimlendirir."""
    lines = [
        "### Uretici Bazli Kirilim\n",
        "| Uretici | n | Accuracy | Macro-F1 |",
        "|---------|---|----------|----------|",
    ]
    for gen, m in by_generator.items():
        lines.append(f"| {gen} | {m['n']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")
    return "\n".join(lines)
