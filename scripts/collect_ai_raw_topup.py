"""Cok-ureticili genelleme takviyesi: GPT-4o-mini ile ek ai_raw verisi uretir.

Amac: ai_raw sinifi su ana kadar SADECE Qwen2.5-7B-Instruct'ten (transformers
saglayicisi) uretildi. Canlı testte bulundu ki model, genel bir "AI mi yazdi"
kavrami degil, Qwen'in spesifik kelime-dagilim imzasini ogrenmis -- Claude'un
yazdigi bambaska bir AI metni bile %97+ "human" cikiyordu (bkz. proje notlari,
2026-07-19, kl_div_word_freq/entropy_2gram ozelliklerinin human bolgesine dustugu
bulundu). Cozum: ikinci, mimari olarak farkli bir uretici (GPT-4o-mini) ile
mevcut Qwen havuzunun USTUNE (yerine degil) yeni ai_raw ornekleri eklemek.

KRITIK -- uzunluk kalibrasyonu: llm_generators.py'deki varsayilan hedef uzunluk
(_TARGET_LEN_MEAN=1750 kelime) Qwen'in hic ulasamadigi eski/asirasyonel bir
degerdir; gercek cozum insan verisini ~650 kelimeye rechunk etmekti (bkz. proje
notlari). GPT-4o-mini Qwen'den farkli olarak uzunluk talimatina gercekten
uyabilir -- varsayilanla cagrilirsa insan/Qwen ile YENİ bir uzunluk uyumsuzlugu
yaratip cozulen confound'u geri getirebilir. Bu yuzden bu script
configs/data_sources.yaml -> cross_generator_topup bolumundeki KALIBRE EDILMIS
(insanin gercek dagilimina -- ort=650/std=254 -- hizali) target_len_* degerlerini
generate_batch'e acikca gecirir, modulun varsayilanini KULLANMAZ.

Cikti: data/raw/ai_raw_openai/ai_raw_openai.jsonl -- mevcut data/raw/ai_raw/ai_raw.jsonl'e
DOKUNMAZ, ayri dosya. Preprocess/build_features asamasinda kisa-pilot verisiyle
ayni yontemle (interim dosyalarina birlestirme) ana havuza katilacak.

Checkpoint/resume: ana collect_data.py ile ayni mantik -- yarida kesilirse ayni
komutla devam edilebilir.

Kullanim:
    python scripts/collect_ai_raw_topup.py
    python scripts/collect_ai_raw_topup.py --target-count 500  # varsayilani (1000) override et
"""
from __future__ import annotations

import argparse

from humanai_detect.config import PROJECT_ROOT, get_api_key, load_yaml
from humanai_detect.data_collection import llm_generators
from humanai_detect.data_collection.schemas import RawSample
from humanai_detect.utils.io import read_jsonl


def collect_topup(paths_cfg: dict, data_sources_cfg: dict, target_count: int | None = None) -> list[RawSample]:
    topup_cfg = data_sources_cfg["cross_generator_topup"]
    provider = topup_cfg["provider"]
    target_count = target_count if target_count is not None else topup_cfg["target_count"]

    provider_cfg = data_sources_cfg["llm_generators"][provider]
    prompts = llm_generators.load_prompts(PROJECT_ROOT / data_sources_cfg["llm_generators"]["prompts_file"])
    # Global rate_limit yerine topup_cfg'deki gercek hesap-seviyesi RPM'i kullan
    # (bkz. topup_cfg icindeki 2026-07-19 notu -- OpenAI hesabinin gercek limiti 10 RPM cikti).
    rate_limit = {
        **(data_sources_cfg.get("rate_limit") or {}),
        "requests_per_minute": topup_cfg.get("requests_per_minute", 9),
        "max_retries": topup_cfg.get("max_retries", 5),
    }

    out_path = PROJECT_ROOT / paths_cfg["raw_dir"] / f"ai_raw_{provider}" / f"ai_raw_{provider}.jsonl"

    existing: list[RawSample] = []
    if out_path.exists():
        existing = [RawSample(**r) for r in read_jsonl(out_path)]
        if len(existing) >= target_count:
            print(f"[topup] {len(existing)} ornek zaten mevcut, atlanıyor.")
            return existing[:target_count]
        print(f"[topup] {len(existing)} mevcut ornek yuklendi, {target_count - len(existing)} daha uretilecek.")

    new_samples = llm_generators.generate_batch(
        prompts,
        provider,
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
                         help="configs/data_sources.yaml -> cross_generator_topup.target_count degerini override eder")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    data_sources_cfg = load_yaml("data_sources")

    samples = collect_topup(paths_cfg, data_sources_cfg, target_count=args.target_count)
    print(f"[topup] Tamamlandi: {len(samples)} ornek.")


if __name__ == "__main__":
    main()
