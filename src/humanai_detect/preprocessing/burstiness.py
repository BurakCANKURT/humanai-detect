"""Cumle uzunlugu varyansina dayali burstiness hesaplamasi."""

from __future__ import annotations

import math
import statistics


def compute_burstiness(sentence_lengths: list[int]) -> float:
    """Goh-Barabasi burstiness parametresini hesaplar: B = (sigma - mu) / (sigma + mu).

    Deger araligi [-1, 1]:
      B ~ +1 : patlamali (bursty) — duzensiz, degisken cumle uzunluklari (insan yazmaya benzer)
      B ~ -1 : duzgun (regular) — monoton, benzer uzunlukta cumleler (AI yazmaya benzer)
      B ~ 0  : rasgele (Poisson)

    Tek cumlede (veya hic cumle yoksa) varyans tanimsizdir; bu durumda NaN dondurulur
    (sabit 0.0 yerine) -- 0.0, "duzenli/AI-benzeri" anlamina gelen gecerli bir deger oldugu
    icin, bilgi eksikligini bununla karistirmamak gerekir. Downstream standardizasyon (fusion/
    standardize.py) NaN'i sutun bazinda goz ardi edip satir bazinda ortalamaya impute eder.

    Kaynak: Goh & Barabasi (2008), EPL 81(4).
    """
    if len(sentence_lengths) < 2:
        return math.nan
    mu = statistics.mean(sentence_lengths)
    sigma = statistics.stdev(sentence_lengths)
    denom = sigma + mu
    if denom == 0.0:
        return 0.0
    return (sigma - mu) / denom
