"""Istatistiksel stilometrik metrikler: n-gram entropisi, KL uzakligi, entropy drop."""

from __future__ import annotations

import math
from collections import Counter


def ngram_entropy(tokens: list[str], n: int) -> float:
    """n-gram dagiliminin Shannon entropisini bit cinsinden hesaplar (Miller-Madow duzeltmeli).

    Yuksek entropi -> cesitli n-gram kullanimi (dogal dile benzer).
    Dusuk entropi -> tekrara dayali, ogrenilebilir yapi (AI ciktisina benzer).

    Plug-in Shannon tahmini kucuk orneklemde (N kucukken, kisa metinlerde oldugu gibi)
    sistematik olarak dusuk cikar; Miller-Madow duzeltmesi (K-1)/(2N ln2) bu yanliligi
    telafi eder. N buyudukce (uzun metin) katki ihmal edilebilir hale gelir, yani ayni
    formul hem kisa hem uzun metinde tutarli kalir.
    """
    if len(tokens) < n or n < 1:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    correction = (len(counts) - 1) / (2 * total * math.log(2))
    return entropy + correction


def kl_divergence_word_freq(
    sample_freqs: dict[str, float],
    reference_freqs: dict[str, float],
) -> float:
    """KL(P_sample || P_reference) — kelime frekansi dagilimlarinin uzakligi.

    sample_freqs ve reference_freqs toplam 1'e normalize edilmis olmali.
    reference'ta olmayan kelimeler icin epsilon smoothing uygulanir.
    """
    eps = 1e-10
    kl = 0.0
    for word, p in sample_freqs.items():
        if p <= 0:
            continue
        q = reference_freqs.get(word, eps)
        kl += p * math.log(p / q)
    return max(0.0, kl)  # sayisal hatadan negatif olmamasi icin


def token_entropy_drop_score(raw_entropy: float, humanized_entropy: float) -> float:
    """Ham-AI ile humanized metin arasindaki unigram entropi dususunu olcer.

    Pozitif deger: humanize islemi entropi kaybettirmis (tespit edilebilir iz).
    """
    return raw_entropy - humanized_entropy
