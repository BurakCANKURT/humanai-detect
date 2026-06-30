"""Her metin icin uretilen ikincil aciklanabilirlik skorlari.

Siniflandirici karar ciktisina ek olarak analistin yorumlayabilecegi,
ozellik-bazli sapma gostergelerini saglar.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def llm_distance_score(
    sample_features: dict[str, float],
    human_reference: dict[str, Any],
) -> float:
    """Ornegin insan referans dagilimina olan Mahalanobis uzakligini hesaplar.

    human_reference sozlugu:
        'mean'  : dict[str, float] — referans ortalama
        'std'   : dict[str, float] — referans standart sapma

    Ortak ozelliklerin standardize edilmis uzakliklarinin L2 normunu dondurur.
    Yuksek deger -> ornek insan yazisindan cok sapma gosteriyor.
    """
    common = [k for k in sample_features if k in human_reference.get("mean", {})]
    if not common:
        return float("nan")

    ref_mean = human_reference["mean"]
    ref_std = human_reference.get("std", {})

    sq_sum = 0.0
    for k in common:
        std = ref_std.get(k, 1.0) or 1.0
        z = (sample_features[k] - ref_mean[k]) / std
        if math.isfinite(z):
            sq_sum += z ** 2

    return math.sqrt(sq_sum / len(common))


def anomaly_heatmap(
    sample_features: dict[str, float],
    reference_stats: dict[str, Any],
) -> dict[str, float]:
    """Ozellik bazli z-score sapma haritasini hesaplar.

    reference_stats:
        'mean' : dict[str, float]
        'std'  : dict[str, float]

    Donus: {feature: z_score} — |z| > 2 anlamli sapma isaretler.
    Pozitif z: ornek referanstan yuksek; negatif: dusuk.
    """
    ref_mean = reference_stats.get("mean", {})
    ref_std = reference_stats.get("std", {})
    heatmap: dict[str, float] = {}

    for k, v in sample_features.items():
        mu = ref_mean.get(k)
        sigma = ref_std.get(k, 1.0) or 1.0
        if mu is None or not math.isfinite(v):
            heatmap[k] = float("nan")
        else:
            heatmap[k] = (v - mu) / sigma

    return heatmap


def conjunction_deviation_index(sample_density: float, reference_density: float) -> float:
    """Baglac yogunlugundaki bagil sapmani dondurur.

    0 -> referansla ayni; +/- -> insandan ne kadar uzak.
    Ayni hesap features.discourse.conjunction_deviation_index'te de var;
    burasi tek-ornek skorlamasi icin kolaylik saglar.
    """
    if reference_density == 0.0:
        return 0.0
    return (sample_density - reference_density) / reference_density


def entropy_drop_score(raw_entropy: float, candidate_entropy: float) -> float:
    """Ham-AI entropisinden aday metnin entropi farkini dondurur.

    Pozitif: humanize edilmis metin ham-AI'dan daha dusuk entropili (tespit izleri var).
    Ayni hesap features.statistical.token_entropy_drop_score'da da var.
    """
    return raw_entropy - candidate_entropy


def syntactic_shift_score(human_depth: float, candidate_depth: float) -> float:
    """Adayin ortalama dependency derinliginin insan referansindan farkini dondurur.

    Negatif: aday insandan daha sığ (AI tipik davranisi).
    Ayni hesap features.syntactic.syntactic_depth_shift'te de var.
    """
    return candidate_depth - human_depth


def compile_secondary_scores(
    sample_id: str,
    sample_features: dict[str, float],
    human_reference: dict[str, Any],
    model_label: str,
    model_confidence: float,
) -> dict[str, Any]:
    """Tek bir ornek icin tum ikincil skorlari birlestirip sozluge paketler.

    Donus JSON'a dogrudan donusturulebilir.
    """
    return {
        "sample_id": sample_id,
        "model_label": model_label,
        "model_confidence": round(model_confidence, 4),
        "llm_distance": round(llm_distance_score(sample_features, human_reference), 4),
        "anomaly_heatmap": {
            k: round(v, 4) for k, v in anomaly_heatmap(sample_features, human_reference).items()
            if math.isfinite(v)
        },
    }
