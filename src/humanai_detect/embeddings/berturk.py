"""BERTurk embedding cikarimi (dbmdz/bert-base-turkish-cased, 768 boyut)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._encoder import embed_batch


def embed_berturk(
    texts: list[str],
    model_id: str = "dbmdz/bert-base-turkish-cased",
    pooling: str = "cls",
    max_length: int = 512,
    batch_size: int = 16,
    device: str = "auto",
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Metin listesi icin BERTurk embedding'lerini cikarir.

    Donus: float32 ndarray, sekil [N, 768].
    """
    return embed_batch(
        texts,
        model_id=model_id,
        pooling=pooling,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
    )
