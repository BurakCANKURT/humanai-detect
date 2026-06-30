"""Turkce okunabilirlik formulleri: Atesman (1997) ve Bezirci-Yilmaz (2010)."""

from __future__ import annotations

_TURKISH_VOWELS = frozenset("aeıioöuüAEIİOÖUÜ")


def _syllable_count(word: str) -> int:
    """Turkce kelimede hece sayisi ~ sesli harf sayisi (en az 1)."""
    return max(1, sum(1 for c in word if c in _TURKISH_VOWELS))


def _mean_syllables_per_word(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(_syllable_count(t) for t in tokens) / len(tokens)


def _mean_words_per_sentence(tokens: list[str], sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    return len(tokens) / len(sentences)


def atesman_score(sentences: list[str], tokens: list[str]) -> float:
    """Atesman (1997) Turkce okunabilirlik skoru.

    OO = 198.825 - 40.175 * HB - 2.610 * KO
    HB = ortalama hece/kelime, KO = ortalama kelime/cumle.

    Yuksek skor -> kolay okunur (~60-100); dusuk skor -> zor okunur (<30).
    """
    if not sentences or not tokens:
        return 0.0
    hb = _mean_syllables_per_word(tokens)
    ko = _mean_words_per_sentence(tokens, sentences)
    return 198.825 - 40.175 * hb - 2.610 * ko


def bezirci_yilmaz_score(sentences: list[str], tokens: list[str]) -> float:
    """Bezirci-Yilmaz (2010) Turkce okunabilirlik skoru.

    Atesman formulunu dogrulayan ve yeniden test eden calisma; ayni katsayilari kullanir.
    OD = 198.825 - 40.175 * OHU - 2.610 * OKU
    """
    if not sentences or not tokens:
        return 0.0
    hb = _mean_syllables_per_word(tokens)
    ko = _mean_words_per_sentence(tokens, sentences)
    return 198.825 - 40.175 * hb - 2.610 * ko
