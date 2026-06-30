"""Cumle bolme ve tokenizasyon — linguistic.analyze uzerinden Stanza kullanir."""

from __future__ import annotations

from .linguistic import analyze


def split_sentences(text: str) -> list[str]:
    """Metni Stanza ile Turkce cumlelere boler."""
    return analyze(text)["sentences"]


def tokenize(text: str) -> list[str]:
    """Metni Stanza ile tokenize eder; tum cumlelerden duz token listesi dondurur."""
    return analyze(text)["tokens"]
