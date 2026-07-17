"""Ozellik matrisi standardizasyonu (z-score veya robust)."""

from __future__ import annotations

import numpy as np


def zscore_standardize(
    features: np.ndarray,
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardizasyonu: (X - mean) / std.

    train_mask verilirse mean/std sadece egitim orneklerinden hesaplanir
    (veri sizintisini onler); tum satirlar donusturulur.

    NaN degerler (orn. tek-cumlelik metinde tanimsiz burstiness) mean/std hesabinda
    goz ardi edilir (nanmean/nanstd) -- boylece bir sutunda birkac NaN olmasi o sutunun
    TAMAMINI bozmaz. Sonucta kalan NaN (o satirin kendi degeri eksikse) 0'a (ortalamaya)
    impute edilir.

    Donus: (standardize_edilmis_matrix, mean_vektoru, std_vektoru)
    """
    if train_mask is not None:
        fit_data = features[train_mask]
    else:
        fit_data = features

    mean = np.nanmean(fit_data, axis=0)
    std = np.nanstd(fit_data, axis=0)
    std = np.where(std == 0, 1.0, std)  # sifir std -> bolme hatasi onle

    standardized = (features - mean) / std
    return np.nan_to_num(standardized, nan=0.0), mean, std


def robust_standardize(
    features: np.ndarray,
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust standardizasyon: (X - median) / IQR.

    Aykiri degerlere duyarsizdir; perplexity gibi sag-carpik dagilimlar icin uygundur.
    NaN degerler median/IQR hesabinda goz ardi edilir (nanmedian/nanpercentile), kalan
    NaN satirlar 0'a impute edilir (bkz. zscore_standardize).

    Donus: (standardize_edilmis_matrix, median_vektoru, iqr_vektoru)
    """
    fit_data = features[train_mask] if train_mask is not None else features
    median = np.nanmedian(fit_data, axis=0)
    q75, q25 = np.nanpercentile(fit_data, [75, 25], axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr == 0, 1.0, iqr)
    standardized = (features - median) / iqr
    return np.nan_to_num(standardized, nan=0.0), median, iqr


def standardize(
    features: np.ndarray,
    method: str = "zscore",
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """method='zscore' veya 'robust' ile standardizasyonu dispatch eder."""
    if method == "zscore":
        return zscore_standardize(features, train_mask)
    if method == "robust":
        return robust_standardize(features, train_mask)
    raise ValueError(f"Bilinmeyen standardizasyon yontemi: {method!r}  (zscore | robust)")
