"""Kisa-metin (5-30 kelime) cok-ureticili takviye: GPT-4o-mini ile ek ai_raw_short verisi uretir.

Amac: data/raw/ai_raw_short/ (bkz. collect_short_pilot.py) SU AN %100 Qwen2.5-7B-Instruct --
kisa metinde de collect_ai_raw_topup.py'nin ana-havuzda cozdugu tek-uretici asiri uyum riski
var, sadece hic test edilmedi (bkz. proje notlari, 2026-07-21 GUNCELLEME 9, veri boslugu #2).
Bu script mevcut Qwen-short havuzuna DOKUNMAZ, GPT-4o-mini ile ayri bir havuz ekler.

KRITIK -- uzunluk kalibrasyonu HENUZ DOGRULANMADI: collect_ai_raw_topup.py'nin kullandigi
970-kelime hedefi icin GPT-4o-mini'nin ~%70 eksik-uretim orani PILOTLA olculmustu (bkz. proje
notlari). Bu oranin COK KISA (5-30 kelime) hedefte de gecerli olup olmadigi BILINMIYOR --
configs/data_sources.yaml -> cross_generator_topup_short'taki target_len_* degerleri ilk tahmin
(short_pilot ile ayni, 5-30). TAM 300'u calistirmadan once KUCUK bir pilotla (--target-count 20)
gercek kelime sayisi olculup gerekirse target_len_mean/std yeniden kalibre edilmeli.

Cikti: data/raw/ai_raw_openai_short/ai_raw_openai_short.jsonl -- data/raw/ai_raw_short/'a
DOKUNMAZ, ayri dosya. Preprocess asamasinda (TOPUP_LABELS) islenip sonra kisa havuza katilacak.

Checkpoint/resume: ana collect_data.py ile ayni mantik -- yarida kesilirse ayni komutla devam
edilebilir.

Kullanim:
    python scripts/collect_ai_raw_openai_short.py --target-count 20   # ONCE kucuk pilot/kalibrasyon
    python scripts/collect_ai_raw_openai_short.py                     # sonra tam 300 (config varsayilani)
"""
from __future__ import annotations

import argparse

from humanai_detect.config import PROJECT_ROOT, get_api_key, load_yaml
from humanai_detect.data_collection import llm_generators
from humanai_detect.data_collection.schemas import RawSample
from humanai_detect.utils.io import read_jsonl

PROVIDER = "openai"


def collect_short_topup(paths_cfg: dict, data_sources_cfg: dict, target_count: int | None = None) -> list[RawSample]:
    topup_cfg = data_sources_cfg["cross_generator_topup_short"]
    target_count = target_count if target_count is not None else topup_cfg["target_count"]

    provider_cfg = data_sources_cfg["llm_generators"][PROVIDER]
    prompts = llm_generators.load_prompts(PROJECT_ROOT / data_sources_cfg["llm_generators"]["prompts_file"])
    rate_limit = {
        **(data_sources_cfg.get("rate_limit") or {}),
        "requests_per_minute": topup_cfg.get("requests_per_minute", 8),
        "max_retries": topup_cfg.get("max_retries", 5),
    }

    out_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "ai_raw_openai_short" / "ai_raw_openai_short.jsonl"

    existing: list[RawSample] = []
    if out_path.exists():
        existing = [RawSample(**r) for r in read_jsonl(out_path)]
        if len(existing) >= target_count:
            print(f"[topup-short] {len(existing)} ornek zaten mevcut, atlanıyor.")
            return existing[:target_count]
        print(f"[topup-short] {len(existing)} mevcut ornek yuklendi, {target_count - len(existing)} daha uretilecek.")

    new_samples = llm_generators.generate_batch(
        prompts,
        PROVIDER,
        target_count - len(existing),
        rate_limit=rate_limit,
        checkpoint_path=out_path,
        start_index=len(existing),
        target_len_mean=topup_cfg["target_len_mean"],
        target_len_std=topup_cfg["target_len_std"],
        target_len_min=topup_cfg["target_len_min"],
        target_len_max=topup_cfg["target_len_max"],
        max_concurrency=topup_cfg.get("max_concurrency", 1),
        model=provider_cfg["model"],
        api_key=get_api_key(provider_cfg["api_key_env"]),
    )
    return existing + new_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-count", type=int, default=None,
                         help="configs/data_sources.yaml -> cross_generator_topup_short.target_count degerini override eder")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    data_sources_cfg = load_yaml("data_sources")

    samples = collect_short_topup(paths_cfg, data_sources_cfg, target_count=args.target_count)
    print(f"[topup-short] Tamamlandi: {len(samples)} ornek.")
    if samples:
        word_counts = [len(s.text.split()) for s in samples]
        avg = sum(word_counts) / len(word_counts)
        target_mean = data_sources_cfg["cross_generator_topup_short"]["target_len_mean"]
        print(f"[topup-short] Ortalama kelime sayisi: {avg:.1f} (hedef ort={target_mean})")
        print(f"[topup-short] min={min(word_counts)} max={max(word_counts)}")


if __name__ == "__main__":
    main()
