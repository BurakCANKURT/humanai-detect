"""Sozcuksel stilometrik metrikler: TTR, hapax legomena, kelime uzunlugu, Yule's K."""

from __future__ import annotations

import statistics
from collections import Counter


def type_token_ratio(tokens: list[str]) -> float:
    """Type-Token Ratio: unique kelime sayisi / toplam token sayisi."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def hapax_legomena_ratio(tokens: list[str]) -> float:
    """Sadece bir kez gecen kelimelerin (hapax) toplam tokenlara orani."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / len(tokens)


def mean_word_length(tokens: list[str]) -> float:
    """Karakter cinsinden ortalama kelime uzunlugu."""
    if not tokens:
        return 0.0
    return statistics.mean(len(t) for t in tokens)


def word_length_std(tokens: list[str]) -> float:
    """Kelime uzunluklarinin standart sapmasi."""
    if len(tokens) < 2:
        return 0.0
    return statistics.stdev(len(t) for t in tokens)


def vocabulary_richness_yule_k(tokens: list[str]) -> float:
    """Yule's K kelime zenginligi metrigi.

    K = 10^4 * (M2 - M1) / M1^2
    M1 = N (toplam token sayisi)
    M2 = sum_v(v^2 * f_v), f_v = v frekansiyla gecen tur sayisi

    Dusuk K -> zengin kelime hazinesi; yuksek K -> tekrara dayali yazi.
    """
    if not tokens:
        return 0.0
    n = len(tokens)
    counts = Counter(tokens)
    freq_of_freq = Counter(counts.values())  # {freq_value: how_many_types}
    m2 = sum(v ** 2 * f for v, f in freq_of_freq.items())
    return 1e4 * (m2 - n) / (n ** 2)
