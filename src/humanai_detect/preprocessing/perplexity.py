"""Dil modeli tabanli pseudo-perplexity olcumu (masked LM, BERT tabanli)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

_TOKENIZER: "AutoTokenizer | None" = None
_MODEL: "AutoModelForMaskedLM | None" = None
_LOADED_MODEL_ID: str = ""
_DEVICE = None

_MAX_LENGTH = 512   # BERT maksimum
_CHUNK_SIZE = 64    # Her ileri geciste kac token maskelenir (bellek/hiz dengesi)


def _get_model(model_id: str):
    """Model ve tokenizer'i tembel yukler; model_id degisirse yeniden yukler.

    Model, varsa GPU'ya tasinir -- aksi halde (CUDA yoksa) sessizce CPU'da kalir.
    Bu tasima olmadan Colab GPU ortaminda bile hesaplama CPU hizinda kalirdi.
    """
    global _TOKENIZER, _MODEL, _LOADED_MODEL_ID, _DEVICE
    if _MODEL is None or _LOADED_MODEL_ID != model_id:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _MODEL = AutoModelForMaskedLM.from_pretrained(model_id).to(_DEVICE)
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
    input_ids = encoding["input_ids"].to(_DEVICE)          # [1, seq_len]
    seq_len = input_ids.size(1)

    # [CLS] ve [SEP] tokenlarini atla (indeks 0 ve seq_len-1)
    content_indices = torch.arange(1, seq_len - 1, device=_DEVICE)
    n = content_indices.numel()
    if n == 0:
        return float("inf")

    mask_id = tokenizer.mask_token_id
    true_ids = input_ids[0, content_indices]  # [n] -- GPU'da kalir, .item() yok

    total_log_prob = torch.zeros((), device=_DEVICE)

    # NOT: Onceki surum her token icin .item() ile tek tek GPU<->CPU senkronizasyonu
    # yapiyordu (yuzlerce senkron nokta/ornek) -- bu GPU'da bile CPU hizina yakin
    # kalmasina sebep oluyordu. Burada maskeleme ve sonuc cikarma tamamen tensor
    # islemleriyle (fancy indexing) yapiliyor, dongu icinde tek bir .item() yok.
    with torch.no_grad():
        for chunk_start in range(0, n, _CHUNK_SIZE):
            chunk_idx = content_indices[chunk_start : chunk_start + _CHUNK_SIZE]  # [c]
            c = chunk_idx.numel()
            row_idx = torch.arange(c, device=_DEVICE)

            batch = input_ids.repeat(c, 1)          # [c, seq_len]
            batch[row_idx, chunk_idx] = mask_id     # her satirda bir token maskele

            logits = model(batch).logits            # [c, seq_len, vocab]
            token_logits = logits[row_idx, chunk_idx]           # [c, vocab]
            log_probs = torch.log_softmax(token_logits, dim=-1)

            true_chunk = true_ids[chunk_start : chunk_start + _CHUNK_SIZE]  # [c]
            total_log_prob += log_probs[row_idx, true_chunk].sum()

    return math.exp(-total_log_prob.item() / n)
