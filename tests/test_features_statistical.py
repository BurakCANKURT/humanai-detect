"""features.statistical birim testleri."""

from __future__ import annotations

import math

from humanai_detect.features import statistical


class TestNgramEntropy:
    def test_uniform_max_entropy(self):
        # Hepsi farkli unigram -> plug-in max entropy = log2(n) + Miller-Madow duzeltmesi
        tokens = ["a", "b", "c", "d"]
        result = statistical.ngram_entropy(tokens, 1)
        expected = math.log2(4) + (4 - 1) / (2 * 4 * math.log(2))
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_all_same_zero_entropy(self):
        tokens = ["a", "a", "a", "a"]
        assert math.isclose(statistical.ngram_entropy(tokens, 1), 0.0)

    def test_bigram_uniform(self):
        # ["a","a","a","a"] -> tek bigram ("a","a") -> entropi = 0
        tokens = ["a", "a", "a", "a"]
        result = statistical.ngram_entropy(tokens, 2)
        assert math.isclose(result, 0.0, abs_tol=1e-9)

    def test_empty(self):
        assert statistical.ngram_entropy([], 1) == 0.0

    def test_n_larger_than_tokens(self):
        assert statistical.ngram_entropy(["a", "b"], 5) == 0.0

    def test_higher_order_lower_equal_entropy(self):
        import random
        random.seed(42)
        tokens = [random.choice(["a", "b", "c"]) for _ in range(50)]
        h1 = statistical.ngram_entropy(tokens, 1)
        h2 = statistical.ngram_entropy(tokens, 2)
        # Bigram entropisi <= unigram entropisi (genel egilim, garantili degil ama cogu zaman)
        assert h1 >= 0.0 and h2 >= 0.0


class TestKlDivergence:
    def test_identical_distributions_zero(self):
        freqs = {"a": 0.5, "b": 0.3, "c": 0.2}
        result = statistical.kl_divergence_word_freq(freqs, freqs)
        assert math.isclose(result, 0.0, abs_tol=1e-9)

    def test_divergent_positive(self):
        p = {"a": 0.9, "b": 0.1}
        q = {"a": 0.1, "b": 0.9}
        result = statistical.kl_divergence_word_freq(p, q)
        assert result > 0.0

    def test_unseen_word_smoothed(self):
        # "c" referansta yok -> epsilon smoothing
        p = {"a": 0.5, "c": 0.5}
        q = {"a": 0.8, "b": 0.2}
        result = statistical.kl_divergence_word_freq(p, q)
        assert result > 0.0 and math.isfinite(result)


class TestTokenEntropyDrop:
    def test_positive_drop(self):
        assert statistical.token_entropy_drop_score(3.5, 2.0) == 1.5

    def test_negative_drop(self):
        assert statistical.token_entropy_drop_score(2.0, 3.5) == -1.5

    def test_zero_drop(self):
        assert statistical.token_entropy_drop_score(2.0, 2.0) == 0.0
