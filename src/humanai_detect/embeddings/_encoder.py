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


def _sample_cache_path(cache_dir: Path, model_id: str, pooling: str, text: str) -> Path:
    key = hashlib.md5(f"{model_id}|{pooling}|{text}".encode()).hexdigest()
    return Path(cache_dir) / f"{key}.npy"


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

    cache_dir verilirse ORNEK-BAZINDA (metin basina tek .npy) onbelleklenir -- yeni bir
    veri seti eklendiginde (orn. cok-ureticili takviye) sadece DAHA ONCE GORULMEMIS
    metinler yeniden hesaplanir, degismeyen orneklerin embedding'i diskten okunur.

    (2026-07-19 duzeltmesi: onceki surum TUM metin listesini tek bir hash altinda
    onbellekliyordu -- tek bir yeni ornek eklenince bile onbellek tamamen gecersiz
    oluyordu, bu da build_features.py'nin her calistirmada TUM veri setini sifirdan
    yeniden hesaplamasina yol aciyordu, bkz. proje notlari.)
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    n = len(texts)
    result: list[np.ndarray | None] = [None] * n
    miss_indices: list[int] = []

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(texts):
            cache_path = _sample_cache_path(cache_dir, model_id, pooling, text)
            if cache_path.exists():
                result[i] = np.load(cache_path)
            else:
                miss_indices.append(i)
    else:
        miss_indices = list(range(n))

    if miss_indices:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.eval()
        model.to(device)

        for start in range(0, len(miss_indices), batch_size):
            batch_idx = miss_indices[start : start + batch_size]
            chunk = [texts[i] for i in batch_idx]
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
            embs = _pool(out.last_hidden_state, enc["attention_mask"], pooling).cpu().float().numpy()

            for local_i, global_i in enumerate(batch_idx):
                emb = embs[local_i]
                result[global_i] = emb
                if cache_dir is not None:
                    np.save(_sample_cache_path(cache_dir, model_id, pooling, texts[global_i]), emb)

    return np.vstack(result)
