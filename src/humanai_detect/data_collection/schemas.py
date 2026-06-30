"""Veri toplama asamasinda kullanilan veri yapilari."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Label = Literal["human", "ai_raw", "ai_humanized"]


@dataclass
class RawSample:
    """Toplanan tek bir metin orneği."""

    id: str
    text: str
    label: Label
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
