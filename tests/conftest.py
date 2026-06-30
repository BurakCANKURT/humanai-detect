"""Ortak pytest fixture'lari: ucl sinifin birer ornek metni."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_human_text() -> str:
    return (
        "Bu calismada, Turkce metinlerde stilometrik ozelliklerin sinif ayrimina "
        "katkisi incelenmistir. Yontem, veri toplama ve degerlendirme asamalarini icerir."
    )


@pytest.fixture
def sample_ai_raw_text() -> str:
    return (
        "Bu arastirma, yapay zeka tarafindan uretilen metinlerin tespiti uzerine "
        "odaklanmaktadir. Onerilen yontem, stilometri ve embedding tabanli ozellikleri birlestirir."
    )


@pytest.fixture
def sample_ai_humanized_text() -> str:
    return (
        "Bu calisma yapay zeka uretimi metinlerin nasil tespit edilebilecegini ele aliyor. "
        "Onerdigimiz yaklasim, dilbilimsel ozellikler ile derin temsillerin bir araya gelmesine dayaniyor."
    )
