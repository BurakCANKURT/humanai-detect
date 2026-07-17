"""Asama 2 on isleme modullerinin birim testleri.

Stanza ve transformers model indirmesi gerektiren testler atlanir (CI-friendly);
sadece saf-Python ve mock bazli testler kosulsuz calisir.
"""

from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# cleaning.py
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_strips_leading_trailing_whitespace(self):
        from humanai_detect.preprocessing.cleaning import clean_text
        assert clean_text("  merhaba  ") == "merhaba"

    def test_normalizes_multiple_spaces(self):
        from humanai_detect.preprocessing.cleaning import clean_text
        assert clean_text("kelime1   kelime2") == "kelime1 kelime2"

    def test_collapses_excessive_newlines(self):
        from humanai_detect.preprocessing.cleaning import clean_text
        result = clean_text("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_removes_control_chars(self):
        from humanai_detect.preprocessing.cleaning import clean_text
        # \x07 (BEL) kontrol karakteri
        assert "\x07" not in clean_text("merhaba\x07dunya")

    def test_nfc_normalization(self):
        from humanai_detect.preprocessing.cleaning import clean_text
        import unicodedata
        # NFD formundaki 'a + combining grave' -> NFC 'à'
        nfd = "à"
        nfc = unicodedata.normalize("NFC", nfd)
        assert clean_text(nfd) == nfc

    def test_empty_string(self):
        from humanai_detect.preprocessing.cleaning import clean_text
        assert clean_text("") == ""

    def test_pdf_hyphen_linebreak(self):
        from humanai_detect.preprocessing.cleaning import clean_text
        # PDF'den gelen "kel-\nime" -> "kelime"
        result = clean_text("kel-\nime")
        assert result == "kelime"


# ---------------------------------------------------------------------------
# burstiness.py
# ---------------------------------------------------------------------------

class TestComputeBurstiness:
    def test_uniform_lengths_near_minus_one(self):
        from humanai_detect.preprocessing.burstiness import compute_burstiness
        # Tamamen esit uzunluklar: sigma=0, B = (0-mu)/(0+mu) = -1
        result = compute_burstiness([10, 10, 10, 10])
        assert math.isclose(result, -1.0)

    def test_single_element_returns_nan(self):
        from humanai_detect.preprocessing.burstiness import compute_burstiness
        # Varyans tek elemanla tanimsiz -- sabit 0.0 (yanlislikla "duzenli" anlamina gelir)
        # yerine NaN donmeli, boylece eksik bilgi gercek bir degerle karistirilmaz.
        assert math.isnan(compute_burstiness([5]))

    def test_empty_returns_nan(self):
        from humanai_detect.preprocessing.burstiness import compute_burstiness
        assert math.isnan(compute_burstiness([]))

    def test_varied_lengths_positive(self):
        from humanai_detect.preprocessing.burstiness import compute_burstiness
        # Cok degisken uzunluklar pozitif B vermeli
        result = compute_burstiness([1, 100, 2, 200, 3, 150])
        assert result > 0.0

    def test_range_minus_one_to_one(self):
        from humanai_detect.preprocessing.burstiness import compute_burstiness
        for lengths in [[5, 5, 5], [1, 10, 100], [3, 7, 2, 8]]:
            b = compute_burstiness(lengths)
            assert -1.0 <= b <= 1.0


# ---------------------------------------------------------------------------
# schemas.py
# ---------------------------------------------------------------------------

class TestProcessedSample:
    def test_default_fields(self):
        from humanai_detect.preprocessing.schemas import ProcessedSample
        s = ProcessedSample(
            id="test_001",
            text="ham metin",
            cleaned_text="ham metin",
            label="human",
            source="test",
        )
        assert s.token_count == 0
        assert s.sentences == []
        assert s.perplexity == 0.0
        assert s.burstiness == 0.0

    def test_dataclass_asdict(self):
        from dataclasses import asdict
        from humanai_detect.preprocessing.schemas import ProcessedSample
        s = ProcessedSample(
            id="x", text="t", cleaned_text="t", label="ai_raw", source="llama",
            tokens=["a", "b"], token_count=2
        )
        d = asdict(s)
        assert d["token_count"] == 2
        assert d["tokens"] == ["a", "b"]
