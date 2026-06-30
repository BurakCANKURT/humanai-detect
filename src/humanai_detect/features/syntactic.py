"""Sozdizimsel stilometrik metrikler: dependency derinligi, POS dagilimi."""

from __future__ import annotations

import statistics
from collections import Counter


def _word_depths(dep_parse: list[dict]) -> list[int]:
    """Her kelimenin kok dugume (head=0) olan hop uzakligini hesaplar."""
    head_map = {w["id"]: w["head"] for w in dep_parse}

    def depth(word_id: int, visited: set | None = None) -> int:
        if visited is None:
            visited = set()
        if word_id in visited:  # dongusel bag — kir
            return 0
        visited.add(word_id)
        parent = head_map.get(word_id, 0)
        if parent == 0:
            return 0
        return 1 + depth(parent, visited)

    return [depth(w["id"]) for w in dep_parse]


def mean_dependency_depth(dep_parse: list[dict]) -> float:
    """Ortalama dependency agac derinligi.

    Yuksek deger -> daha karmasik sozdizimsel yapilar.
    AI metinleri genellikle daha yuzeysel (dusuk) derinlikle calisir.
    """
    if not dep_parse:
        return 0.0
    return statistics.mean(_word_depths(dep_parse))


def dependency_depth_std(dep_parse: list[dict]) -> float:
    """Dependency agac derinliklerinin standart sapmasi."""
    if len(dep_parse) < 2:
        return 0.0
    return statistics.stdev(_word_depths(dep_parse))


def pos_distribution(pos_tags: list[tuple[str, str]]) -> dict[str, float]:
    """POS etiketlerinin normalize edilmis frekans dagilimini dondurur.

    Cikti: {"NOUN": 0.32, "VERB": 0.18, ...}  — oranlar toplam 1'e esittir.
    """
    if not pos_tags:
        return {}
    total = len(pos_tags)
    counts = Counter(pos for _, pos in pos_tags)
    return {pos: count / total for pos, count in counts.items()}


def syntactic_depth_shift(
    human_depths: list[float],
    candidate_depths: list[float],
) -> float:
    """Adayin ortalama dep. derinliginin insan referansindan farkini dondurur.

    Negatif deger: aday, insandan daha sığ (AI tipik davranisi).
    """
    if not human_depths or not candidate_depths:
        return 0.0
    return statistics.mean(candidate_depths) - statistics.mean(human_depths)
