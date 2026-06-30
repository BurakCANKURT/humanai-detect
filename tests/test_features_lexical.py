"""features.lexical birim testleri."""

from __future__ import annotations

import math

from humanai_detect.features import lexical


class TestTypetokenRatio:
    def test_all_unique(self):
        assert lexical.type_token_ratio(["a", "b", "c"]) == 1.0

    def test_all_same(self):
        assert lexical.type_token_ratio(["a", "a", "a"]) == pytest.approx(1 / 3)

    def test_empty(self):
        assert lexical.type_token_ratio([]) == 0.0

    def test_range(self):
        result = lexical.type_token_ratio(["bu", "bir", "bu", "metin"])
        assert 0.0 < result <= 1.0


class TestHapaxLegomenaRatio:
    def test_all_hapax(self):
        assert lexical.hapax_legomena_ratio(["a", "b", "c"]) == 1.0

    def test_no_hapax(self):
        assert lexical.hapax_legomena_ratio(["a", "a", "b", "b"]) == 0.0

    def test_mixed(self):
        # "a" 2x, "b" 1x, "c" 1x -> hapax: b, c -> 2/4
        result = lexical.hapax_legomena_ratio(["a", "a", "b", "c"])
        assert math.isclose(result, 0.5)

    def test_empty(self):
        assert lexical.hapax_legomena_ratio([]) == 0.0


class TestMeanWordLength:
    def test_basic(self):
        result = lexical.mean_word_length(["ab", "cde", "f"])
        assert math.isclose(result, (2 + 3 + 1) / 3)

    def test_empty(self):
        assert lexical.mean_word_length([]) == 0.0


class TestWordLengthStd:
    def test_uniform_zero_std(self):
        assert math.isclose(lexical.word_length_std(["ab", "cd", "ef"]), 0.0)

    def test_single_token(self):
        assert lexical.word_length_std(["kelime"]) == 0.0

    def test_varied(self):
        assert lexical.word_length_std(["a", "bbb"]) > 0.0


class TestYuleK:
    def test_empty(self):
        assert lexical.vocabulary_richness_yule_k([]) == 0.0

    def test_all_unique_gives_lower_k(self):
        all_unique = [str(i) for i in range(20)]
        all_same = ["a"] * 20
        k_unique = lexical.vocabulary_richness_yule_k(all_unique)
        k_same = lexical.vocabulary_richness_yule_k(all_same)
        # Tekrara dayali metin daha yuksek K
        assert k_same > k_unique


import pytest  # noqa: E402  (sinif bloklari import'tan once tanimlandi, pytest fixture icin en sona)
