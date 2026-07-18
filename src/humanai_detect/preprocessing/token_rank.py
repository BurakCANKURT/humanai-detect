"""Causal LM tabanli token-rank (GLTR-tarzi) ozellikler."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

_TOKENIZER: "AutoTokenizer | None" = None
_MODEL: "AutoModelForCausalLM | None" = None
_LOADED_MODEL_ID: str = ""
_DEVICE = None

_MAX_LENGTH = 512   # causal LM baglam siniri

_EMPTY_STATS = {
    "mean_token_rank": 0.0,
    "frac_rank_top1": 0.0,
    "frac_rank_top5": 0.0,
    "frac_rank_top10": 0.0,
    "rank_entropy": 0.0,
}


def _get_model(model_id: str):
    """Model ve tokenizer'i tembel yukler; model_id degisirse yeniden yukler.

    perplexity.py::_get_model ile ayni desen -- GPU varsa modeli tasir, yoksa CPU'da kalir.
    """
    global _TOKENIZER, _MODEL, _LOADED_MODEL_ID, _DEVICE
    if _MODEL is None or _LOADED_MODEL_ID != model_id:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _MODEL = AutoModelForCausalLM.from_pretrained(model_id).to(_DEVICE)
        _MODEL.eval()
        _LOADED_MODEL_ID = model_id
    return _TOKENIZER, _MODEL


def compute_token_rank_stats(text: str, model_id: str) -> dict[str, float]:
    """Her token'in, causal LM'in bir onceki baglamdaki tahmin siralamasindaki (rank) yerini olcer.

    Tek ileri-gecis yeterli: causal LM her pozisyon icin bir sonraki token'in tam olasilik
    dagilimini teacher-forcing ile uretir -- perplexity.py'deki maskeli-LM yontemi gibi
    token-token yeniden ileri-gecis gerektirmez, bu yuzden hesapca daha ucuzdur.

    Rank = gercek token'dan daha yuksek logit'e sahip token sayisi + 1 (1 = modelin en cok
    bekledigi kelime). AI metni dusuk (tahmin edilebilir) rank'lara yigilma egilimindedir;
    insan metninde rank dagilimi daha genis ve carpiktir (bursty kelime seçimi).

    Uzun metinler icin ilk max_length token alinir (causal LM baglam siniri).
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
    input_ids = encoding["input_ids"].to(_DEVICE)
    seq_len = input_ids.size(1)

    if seq_len < 2:
        return dict(_EMPTY_STATS)

    with torch.no_grad():
        with torch.autocast(device_type=_DEVICE.type, dtype=torch.float16, enabled=(_DEVICE.type == "cuda")):
            logits = model(input_ids).logits  # [1, seq_len, vocab]

    # pozisyon i'nin logit'i pozisyon i+1'deki gercek token'i tahmin eder (teacher forcing)
    pred_logits = logits[0, :-1, :].float()   # [seq_len-1, vocab]
    true_ids = input_ids[0, 1:]                # [seq_len-1]

    true_logits = pred_logits.gather(1, true_ids.unsqueeze(1)).squeeze(1)          # [seq_len-1]
    ranks = (pred_logits > true_logits.unsqueeze(1)).sum(dim=1) + 1                # [seq_len-1], 1-indexed
    ranks_f = ranks.float()

    mean_rank = ranks_f.mean().item()
    frac_top1 = (ranks == 1).float().mean().item()
    frac_top5 = (ranks <= 5).float().mean().item()
    frac_top10 = (ranks <= 10).float().mean().item()

    # log2-rank'i 10 esit-genislikte kovaya ayirip histogram (Shannon) entropisi hesaplanir --
    # rank dagiliminin ne kadar "yayildigini" ozetler (dusuk entropi = rank'lar dar bir bantta yigilmis).
    log_ranks = torch.log2(ranks_f)
    max_log_rank = math.log2(pred_logits.size(1))
    bucket_width = max_log_rank / 10.0
    buckets = torch.clamp((log_ranks / bucket_width).long(), max=9)
    counts = torch.bincount(buckets, minlength=10).float()
    probs = counts / counts.sum()
    nonzero = probs[probs > 0]
    entropy = -(nonzero * torch.log2(nonzero)).sum().item()

    return {
        "mean_token_rank": mean_rank,
        "frac_rank_top1": frac_top1,
        "frac_rank_top5": frac_top5,
        "frac_rank_top10": frac_top10,
        "rank_entropy": entropy,
    }
