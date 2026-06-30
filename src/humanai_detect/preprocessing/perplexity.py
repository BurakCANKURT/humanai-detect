"""Dil modeli tabanli pseudo-perplexity olcumu (masked LM, BERT tabanli)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

_TOKENIZER: "AutoTokenizer | None" = None
_MODEL: "AutoModelForMaskedLM | None" = None
_LOADED_MODEL_ID: str = ""

_MAX_LENGTH = 512   # BERT maksimum
_CHUNK_SIZE = 64    # Her ileri geciste kac token maskelenir (bellek/hiz dengesi)


def _get_model(model_id: str):
    """Model ve tokenizer'i tembel yukler; model_id degisirse yeniden yukler."""
    global _TOKENIZER, _MODEL, _LOADED_MODEL_ID
    if _MODEL is None or _LOADED_MODEL_ID != model_id:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _MODEL = AutoModelForMaskedLM.from_pretrained(model_id)
        _MODEL.eval()
        _LOADED_MODEL_ID = model_id
    return _TOKENIZER, _MODEL


def compute_perplexity(text: str, model_id: str) -> float:
    """Masked LM pseudo-perplexity hesaplar (Salazar et al. 2020 yontemi).

    Her token sirayla maskelenir, modelin o token icin verdigi log-olasilik
    toplanir ve geometrik ortalama alinir: PPL = exp(-1/N * sum(log P(xi | x_mask_i))).

    Uzun metinler icin ilk max_length token alinir (BERT siniri).
    Bellek tasarrufu icin maskeleme islemi chunk_size'lik gruplar halinde yapilir.
    """
    import torch

    tokenizer, model = _get_model(model_id)

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_LENGTH,
        add_special_tokens=True,
    )
    input_ids = encoding["input_ids"]          # [1, seq_len]
    seq_len = input_ids.size(1)

    # [CLS] ve [SEP] tokenlarini atla (indeks 0 ve seq_len-1)
    content_indices = list(range(1, seq_len - 1))
    if not content_indices:
        return float("inf")

    total_log_prob = 0.0
    mask_id = tokenizer.mask_token_id

    with torch.no_grad():
        for chunk_start in range(0, len(content_indices), _CHUNK_SIZE):
            chunk = content_indices[chunk_start : chunk_start + _CHUNK_SIZE]
            # Her satirda bir token maskeli olan batch olustur
            batch = input_ids.repeat(len(chunk), 1)  # [chunk, seq_len]
            true_ids = []
            for row, idx in enumerate(chunk):
                true_ids.append(input_ids[0, idx].item())
                batch[row, idx] = mask_id

            logits = model(batch).logits  # [chunk, seq_len, vocab]
            for row, idx in enumerate(chunk):
                log_probs = torch.log_softmax(logits[row, idx], dim=-1)
                total_log_prob += log_probs[true_ids[row]].item()

    n = len(content_indices)
    return math.exp(-total_log_prob / n)
