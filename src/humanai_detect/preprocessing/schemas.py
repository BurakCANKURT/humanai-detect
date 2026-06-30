"""On isleme ciktisi icin veri yapilari."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from humanai_detect.data_collection.schemas import Label


@dataclass
class ProcessedSample:
    """cleaning -> tokenizasyon -> linguistik analiz -> perplexity -> burstiness sonrasi ornek."""

    id: str
    text: str           # ham metin (data/raw'dan gelen)
    cleaned_text: str   # normalize edilmis metin
    label: Label
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Cumle/token yapisi (Stanza ciktisi)
    sentences: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    pos_tags: list[list[str, str]] = field(default_factory=list)  # [(token, UPOS), ...]
    dep_parse: list[dict] = field(default_factory=list)          # per-word depparse kaydi

    token_count: int = 0
    sentence_count: int = 0

    # Istatistiksel/model tabanli ozellikler (Asama 3 icin hazir)
    perplexity: float = 0.0
    burstiness: float = 0.0
