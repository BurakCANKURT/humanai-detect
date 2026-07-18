"""Asama 2: temizleme, tokenizasyon, POS/dependency, perplexity, burstiness.

Girdi : data/raw/{label}/{label}.jsonl
Cikti : data/interim/{label}/{label}.jsonl

Her RawSample icin:
  1. clean_text       -- unicode norm, gurultu temizligi
  2. linguistic.analyze -- Stanza: cumleler, tokenlar, POS, depparse (tek gecis)
  3. compute_perplexity -- masked-LM pseudo-PPL (dbmdz/bert-base-turkish-cased)
  4. compute_perplexity (2. model) -- capraz-model perplexity orani icin (Binoculars-tarzi)
  5. compute_token_rank_stats -- causal LM ile GLTR-tarzi rank istatistikleri
  6. compute_burstiness -- Goh-Barabasi B parametresi
  7. token_count filtresi -- [min_tokens, max_tokens] araligi disindakiler atilir
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.data_collection.schemas import RawSample
from humanai_detect.preprocessing import (
    ProcessedSample,
    analyze,
    clean_text,
    compute_burstiness,
    compute_perplexity,
    compute_token_rank_stats,
)
from humanai_detect.utils.io import read_jsonl, write_jsonl

LABELS = ["human", "ai_raw", "ai_humanized"]
SHORT_LABELS = ["human_short", "ai_raw_short", "ai_humanized_short"]


_STANZA_WORD_LIMIT = 500  # Stanza depparse icin max kelime (hiz siniri)


def _fast_token_estimate(text: str) -> int:
    """Bosluk ile yaklasik token sayisi tahmini (Stanza'dan onceki hizli filtre)."""
    return len(text.split())


def _truncate_for_stanza(text: str, limit: int = _STANZA_WORD_LIMIT) -> str:
    """Stanza is yukunu sinirlamak icin metni ilk `limit` kelimeye kirpar.

    Ayrica tek cumleler cok uzunsa (>100 kelime) Stanza depparse O(n^2) yavaslar;
    bu durumda cumleyi 100 kelimede keserek yeni cumle baslatiriz.
    """
    # Once kelime limitini uygula
    words = text.split()
    if len(words) > limit:
        text = " ".join(words[:limit])

    # Uzun cumleler icin ek koruma: her 100 kelimede nokta ekle
    sentences = text.split(". ")
    result = []
    for sent in sentences:
        sent_words = sent.split()
        if len(sent_words) > 100:
            # 100 kelimelik parcalara bol
            for i in range(0, len(sent_words), 100):
                result.append(" ".join(sent_words[i:i+100]))
        else:
            result.append(sent)
    return ". ".join(result)


def process_sample(
    raw: RawSample,
    preprocessing_cfg: dict,
) -> ProcessedSample | None:
    """Tek bir RawSample'i isler; token siniri disindaysa None dondurur."""
    cleaned = clean_text(raw.text)
    if not cleaned:
        return None

    # Stanza'yi cagirmadan once hizli on filtre (cok kisa/uzun metinleri eler)
    approx_tokens = _fast_token_estimate(cleaned)
    max_tok = preprocessing_cfg["max_tokens"]
    min_tok = preprocessing_cfg["min_tokens"]
    if approx_tokens < min_tok * 0.5:
        return None

    # Stanza: sadece ilk STANZA_WORD_LIMIT kelime (depparse hiz siniri)
    # Perplexity tam metni kullanmaya devam eder
    stanza_text = _truncate_for_stanza(cleaned)
    result = analyze(stanza_text)
    token_count = len(result["tokens"])

    if not (preprocessing_cfg["min_tokens"] <= token_count <= preprocessing_cfg["max_tokens"]):
        return None

    sentence_lengths = [len(s.split()) for s in result["sentences"]]
    burstiness = compute_burstiness(sentence_lengths)

    perplexity = compute_perplexity(cleaned, preprocessing_cfg["perplexity_model_id"])
    perplexity_2 = compute_perplexity(cleaned, preprocessing_cfg["perplexity_ratio_model_id"])
    perplexity_ratio = (
        perplexity / perplexity_2
        if math.isfinite(perplexity) and math.isfinite(perplexity_2) and perplexity_2 > 0
        else 1.0
    )

    rank_stats = compute_token_rank_stats(cleaned, preprocessing_cfg["causal_lm_model_id"])

    # VRAM birikimini onle
    try:
        import torch, gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    return ProcessedSample(
        id=raw.id,
        text=raw.text,
        cleaned_text=cleaned,
        label=raw.label,
        source=raw.source,
        metadata=raw.metadata,
        sentences=result["sentences"],
        tokens=result["tokens"],
        pos_tags=result["pos_tags"],
        dep_parse=result["dep_parse"],
        token_count=token_count,
        sentence_count=len(result["sentences"]),
        perplexity=perplexity,
        burstiness=burstiness,
        perplexity_ratio=perplexity_ratio,
        mean_token_rank=rank_stats["mean_token_rank"],
        frac_rank_top1=rank_stats["frac_rank_top1"],
        frac_rank_top5=rank_stats["frac_rank_top5"],
        frac_rank_top10=rank_stats["frac_rank_top10"],
        rank_entropy=rank_stats["rank_entropy"],
    )


def process_label(label: str, input_dir, output_dir, preprocessing_cfg: dict, limit: int | None = None) -> None:
    in_path = input_dir / label / f"{label}.jsonl"
    if not in_path.exists():
        print(f"[{label}] {in_path} bulunamadi, atlanıyor.")
        return

    raw_samples = [RawSample(**r) for r in read_jsonl(in_path)]
    if limit is not None:
        raw_samples = raw_samples[:limit]
    print(f"[{label}] {len(raw_samples)} ornek yuklendi.")

    out_path = output_dir / label / f"{label}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint: mevcut ciktidan tamamlananlar atlanir
    done_ids: set[str] = set()
    if out_path.exists():
        for rec in read_jsonl(out_path):
            done_ids.add(rec["id"])
        if done_ids:
            print(f"[{label}] {len(done_ids)} ornek checkpoint'ten yuklendi, kalan: {len(raw_samples) - len(done_ids)}")

    skipped = 0
    for i, raw in enumerate(raw_samples, 1):
        if limit is not None and len(done_ids) >= limit:
            break
        if raw.id in done_ids:
            continue
        print(f"  [{i}/{len(raw_samples)}] {raw.id} isleniyor...", flush=True)
        result = process_sample(raw, preprocessing_cfg)
        if result is None:
            skipped += 1
            print(f"  [{i}/{len(raw_samples)}] atildi (token siniri veya bos metin).")
        else:
            # Aninda yaz (her ornek checkpoint gorevi gorur)
            with out_path.open("a", encoding="utf-8") as fp:
                import json
                fp.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            done_ids.add(raw.id)

    total_written = sum(1 for _ in read_jsonl(out_path))
    print(f"[{label}] {total_written} ornek hazir ({skipped} atildi) -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        choices=[*LABELS, *SHORT_LABELS, "all"],
        default="all",
        help="Hangi sinif icin on isleme yapilacak",
    )
    parser.add_argument("--input-dir", default=None, help="data/raw dizini (varsayilan: configs/paths.yaml)")
    parser.add_argument("--output-dir", default=None, help="data/interim dizini (varsayilan: configs/paths.yaml)")
    parser.add_argument("--limit", type=int, default=None, help="Sinif basina en fazla kac ornek islenecek (hizli pilot/benchmark icin)")
    parser.add_argument("--min-tokens", type=int, default=None, help="configs/preprocessing.yaml min_tokens degerini bu calisma icin gecersiz kilar (kisa-pilot verisi icin gerekli)")
    parser.add_argument("--max-tokens", type=int, default=None, help="configs/preprocessing.yaml max_tokens degerini bu calisma icin gecersiz kilar")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    preprocessing_cfg = load_yaml("preprocessing")
    if args.min_tokens is not None:
        preprocessing_cfg["min_tokens"] = args.min_tokens
    if args.max_tokens is not None:
        preprocessing_cfg["max_tokens"] = args.max_tokens

    input_dir = PROJECT_ROOT / (args.input_dir or paths_cfg["raw_dir"])
    output_dir = PROJECT_ROOT / (args.output_dir or paths_cfg["interim_dir"])

    labels = LABELS if args.label == "all" else [args.label]
    for label in labels:
        process_label(label, input_dir, output_dir, preprocessing_cfg, limit=args.limit)


if __name__ == "__main__":
    main()
