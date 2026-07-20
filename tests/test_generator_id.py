"""utils.generator_id birim testleri."""

from __future__ import annotations

import pytest

from humanai_detect.utils.generator_id import infer_generator


class TestInferGenerator:
    @pytest.mark.parametrize(
        "sample_id,expected",
        [
            ("human_rechunked_0000", "human"),
            ("human_doc_file_0012", "human"),
            ("ai_raw_transformers_0001", "qwen"),
            ("ai_humanized_backtranslate_ai_raw_transformers_2153", "qwen"),
            ("ai_raw_openai_0690", "gpt4o_mini"),
            ("ai_humanized_backtranslate_ai_raw_openai_0196", "gpt4o_mini"),
            ("ai_raw_anthropic_0042", "claude_sonnet5"),
            ("ai_humanized_backtranslate_ai_raw_anthropic_0007", "claude_sonnet5"),
        ],
    )
    def test_known_prefixes(self, sample_id, expected):
        assert infer_generator(sample_id) == expected

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError):
            infer_generator("ai_raw_unknown_provider_0001")
