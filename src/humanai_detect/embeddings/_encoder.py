"""Ortak HuggingFace encoder altyapisi — BERTurk ve RoBERTa-TR tarafindan kullanilir."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _pool(hidden: "torch.Tensor", attention_mask: "torch.Tensor", pooling: str) -> "torch.Tensor":
    """Son gizli katmandan tek bir vektor uretir."""
    import torch

    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    if pooling == "max":
        mask = attention_mask.unsqueeze(-1).bool()
        hidden = hidden.masked_fill(~mask, float("-inf"))
        return hidden.max(dim=1).values
    raise ValueError(f"Bilinmeyen pooling stratejisi: {pooling!r}  (cls | mean | max)")


def embed_batch(
    texts: list[str],
    model_id: str,
    pooling: str = "cls",
    max_length: int = 512,
    batch_size: int = 16,
    device: str = "auto",
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Herhangi bir HuggingFace AutoModel encoder ile [N, hidden] embedding matrisi uretir.

    cache_dir verilirse sonuc .npy olarak kaydedilir; ikinci cagirda diskten okunur.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if cache_dir is not None:
        cache_key = hashlib.md5(
            f"{model_id}|{pooling}|{'|'.join(texts)}".encode()
        ).hexdigest()
        cache_path = Path(cache_dir) / f"{cache_key}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    model.to(device)

    parts: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        embs = _pool(out.last_hidden_state, enc["attention_mask"], pooling)
        parts.append(embs.cpu().float().numpy())

    result = np.vstack(parts)

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(cache_path, result)

    return result
