"""RoBERTa-Turkish embedding cikarimi (768 boyut).

Model ID configs/embeddings.yaml -> models.roberta_turkish.model_id ile belirlenir.
Varsayilan: loodos/roberta-base-turkish-mc4 (CC-BY lisansli, HuggingFace'de mevcut).
configs/embeddings.yaml'da 'GroNLP/roberta-base-turkish' yer tutucu olarak tanimli;
gercek calismada bu ID dogrulanmali / guncellenmeli.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._encoder import embed_batch

_DEFAULT_MODEL = "loodos/roberta-base-turkish-mc4"


def embed_roberta_tr(
    texts: list[str],
    model_id: str = _DEFAULT_MODEL,
    pooling: str = "cls",
    max_length: int = 512,
    batch_size: int = 16,
    device: str = "auto",
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Metin listesi icin RoBERTa-Turkish embedding'lerini cikarir.

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
