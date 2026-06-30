"""Confusion matrix, ROC egrisi ve UMAP embedding-space gorsellestirmeleri."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: list[str],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Normalize edilmis confusion matrix gorselini PNG olarak kaydeder."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Confusion matrix kaydedildi -> {out_path}")


def plot_roc_curves(
    y_true: Any,
    y_proba: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str = "ROC Curves (One-vs-Rest)",
) -> None:
    """Her sinif icin OvR ROC egrisi cizer ve PNG olarak kaydeder."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true)
    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, lbl in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{lbl} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] ROC egrisi kaydedildi -> {out_path}")


def plot_umap_projection(
    embeddings: np.ndarray,
    labels: Any,
    out_path: Path,
    label_names: list[str] | None = None,
    title: str = "UMAP Projection",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> None:
    """UMAP ile yuksek boyutlu embedding uzayinin 2B projeksiyonunu kaydeder."""
    import matplotlib.pyplot as plt
    from umap import UMAP

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    proj = reducer.fit_transform(embeddings)

    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(unique_labels)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = (label_names[idx] if label_names and idx < len(label_names) else str(lbl))
        ax.scatter(proj[mask, 0], proj[mask, 1], c=[colors[idx]], label=name, s=20, alpha=0.7)

    ax.set_title(title)
    ax.legend(markerscale=2)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] UMAP projeksiyonu kaydedildi -> {out_path}")
