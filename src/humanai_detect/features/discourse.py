"""Soylem duzeyi metrikler: baglac yogunlugu, fonksiyon kelime orani."""

from __future__ import annotations

# Universal POS etiket kumeleri (UPOS, Stanza ciktisi)
_CONJUNCTION_POS = {"CCONJ", "SCONJ"}
_FUNCTION_WORD_POS = {"DET", "PRON", "ADP", "AUX", "PART", "SCONJ", "CCONJ", "INTJ"}


def conjunction_density(tokens: list[str], pos_tags: list[tuple[str, str]]) -> float:
    """Baglac (CCONJ + SCONJ) kullanim yogunlugu: baglac sayisi / toplam token.

    AI metinleri genellikle daha az varyasyonlu, daha yuksek baglac yogunluguna sahip.
    """
    if not tokens:
        return 0.0
    conj_count = sum(1 for _, pos in pos_tags if pos in _CONJUNCTION_POS)
    return conj_count / len(tokens)


def conjunction_deviation_index(sample_density: float, reference_density: float) -> float:
    """Orneklerin baglac yogunlugunun insan referansindan bagil sapmasini olcer.

    0 -> referansla ayni; pozitif -> insandan daha baglaç-agir; negatif -> daha az.
    reference_density sifirsa 0.0 doner.
    """
    if reference_density == 0.0:
        return 0.0
    return (sample_density - reference_density) / reference_density


def function_word_ratio(tokens: list[str], pos_tags: list[tuple[str, str]]) -> float:
    """Fonksiyon kelimelerin (DET/PRON/ADP/AUX/PART/CONJ/INTJ) toplam tokenlara orani.

    AI metinleri genellikle daha yuksek fonksiyon kelime oranina sahiptir
    (daha basit ve tekrar eden yapisal kaliplar).
    """
    if not tokens:
        return 0.0
    fw_count = sum(1 for _, pos in pos_tags if pos in _FUNCTION_WORD_POS)
    return fw_count / len(tokens)
