"""Metin uzunluguna (token_count) asiri duyarli stilometrik ozellikleri (SHAP top-4)
log-uzunluk uzerinden residualize ederek uzunluk-invaryant hale getirir.

Motivasyon: kl_div_word_freq, entropy_2gram, entropy_3gram, burstiness metin uzunluguna
sistematik olarak duyarli --  n-gram entropisi kucuk orneklemde dusuk kalir (kucuk-ornek
yanliligi), KL sapmasi az kelimeyle daha gurultulu/sisirilmis olur, burstiness az cumleyle
guvenilmez olur (Goh-Barabasi formulunun varyans tahmini kucuk n'de kararsizdir). Bu, modelin
"insan/AI" farkinin yaninda kismen "uzun/kisa metin" ogrenmesine yol acan bir confound'dur
(bkz. proje SHAP analizi: bu 4 ozellik en etkili 4 ozellik cikti, ayrica kalibrasyon uygulandiktan
sonra dahi kisa/OOD metinde model her zaman "human" tahmin etmeye devam etti).

Yontem: her feature icin egitim (train_mask) alt kumesinde log1p(token_count) -> feature
lineer regresyonu fit edilir; sonra TUM orneklerde deger, bu regresyonun ARTIGI (residual)
ile degistirilir: "bu uzunluktaki tipik bir metin icin beklenen degerden sapma". Boylece
uzunlukla dogrudan korele olan bilesen cikarilmis, geriye kalan (varsa) gercek stil farki
sinyali kalir.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LENGTH_SENSITIVE_FEATURES = ["kl_div_word_freq", "entropy_2gram", "entropy_3gram", "burstiness"]


def fit_length_residualizer(
    df: pd.DataFrame,
    token_counts: np.ndarray,
    feature_names: list[str] = LENGTH_SENSITIVE_FEATURES,
    train_mask: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Her feature icin log1p(token_count) -> feature lineer regresyon katsayilarini fit eder.

    train_mask verilirse (orn. held-out haric) SADECE o alt kumeyle fit edilir -- held-out'a
    sizinti onlenir (standardize.py'deki train_mask deseniyle ayni mantik).

    Donus: {feature_name: {"slope": float, "intercept": float}}
    """
    mask = train_mask if train_mask is not None else np.ones(len(df), dtype=bool)
    log_tok = np.log1p(np.asarray(token_counts, dtype=float))

    params: dict[str, dict[str, float]] = {}
    for feat in feature_names:
        if feat not in df.columns:
            continue
        y = df[feat].to_numpy(dtype=float)
        valid = mask & ~np.isnan(y) & ~np.isnan(log_tok)
        if valid.sum() < 10:
            params[feat] = {"slope": 0.0, "intercept": 0.0}
            continue
        slope, intercept = np.polyfit(log_tok[valid], y[valid], 1)
        params[feat] = {"slope": float(slope), "intercept": float(intercept)}
    return params


def apply_length_residualizer_df(
    df: pd.DataFrame,
    token_counts: np.ndarray,
    params: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """fit_length_residualizer ciktisini bir DataFrame'in ilgili sutunlarina uygular."""
    df = df.copy()
    log_tok = np.log1p(np.asarray(token_counts, dtype=float))
    for feat, p in params.items():
        if feat not in df.columns:
            continue
        predicted = p["slope"] * log_tok + p["intercept"]
        df[feat] = df[feat].to_numpy(dtype=float) - predicted
    return df


def apply_length_residualizer_dict(
    feats: dict[str, Any],
    token_count: int,
    params: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Tek bir ornegin (canli cikarim) feature sozlugune ayni donusumu uygular."""
    feats = dict(feats)
    log_tok = float(np.log1p(token_count))
    for feat, p in params.items():
        if feat not in feats or feats[feat] is None:
            continue
        val = feats[feat]
        if isinstance(val, float) and np.isnan(val):
            continue
        predicted = p["slope"] * log_tok + p["intercept"]
        feats[feat] = val - predicted
    return feats
