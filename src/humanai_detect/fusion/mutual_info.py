"""Mutual information tabanli feature secimi ve skorlama."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_mutual_info_scores(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Her ozellik icin sinif etiketiyle mutual information skorunu hesaplar.

    NaN iceren satir/sutunlar ortalama ile doldurulur (imputation).
    Donus: [n_features] seklinde float skor dizisi.
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy="mean")
    X = imp.fit_transform(features)
    return mutual_info_classif(X, labels, random_state=random_state)


def select_by_mutual_info(
    features: np.ndarray,
    labels: np.ndarray,
    top_k: int,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Etiketle en yuksek MI'ya sahip top_k ozelligin indekslerini dondurur.

    Donus: (secilen_indeksler, tum_mi_skorlari)
    """
    scores = compute_mutual_info_scores(features, labels)
    top_k = min(top_k, len(scores))
    indices = np.argsort(scores)[::-1][:top_k]
    return indices, scores


def mi_feature_report(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Ozellik adi ve MI skoru iceren siralı bir DataFrame dondurur (analiz icin)."""
    scores = compute_mutual_info_scores(features, labels)
    df = pd.DataFrame({"feature": feature_names, "mi_score": scores})
    return df.sort_values("mi_score", ascending=False).reset_index(drop=True)
