"""Qwen ANA HAVUZ (uzun-metin, ~2810 ornek) ai_raw'ini GPT-4o-mini PROMPT-TABANLI yeniden
yazimla humanize eder (back-translation DEGIL).

Amac (bkz. proje notlari, 2026-07-25): LOGO-CV'de Qwen-egitim-disi fold'unda ai_raw<->
ai_humanized alt-tur karisikliginin (katı doğruluk sadece %65.6) kok nedeni, Qwen'in ana
havuzdaki ai_humanized'inin hala geri-ceviri (Helsinki-NLP back-translation) ile uretilmis
olmasi bulundu -- geri-ceviri Qwen'in guclu/ekstrem AI-imzasini (dusuk entropi, sig sozdizim)
yeterince silemiyor (ham<->humanized farki, ham<->insan farkinin sadece %5-20'si kadar
kaliyor), bu da iki alt-turu neredeyse ayirt edilemez hale getiriyor. Kisa-metin havuzunda
ayni sorun bulunup GPT-4o-mini prompt-tabanli yeniden yazimla cozulmustu (bkz.
humanize_short_openai.py, humanize_with_openai) -- bu script ayni cozumu ana havuza uygular.

Uzunluk kalibrasyonu: insan ana-havuz ortalamasina (~646 kelime, std~257) hedeflenir --
Qwen ham metninin KENDI uzunlugu DEGIL, projenin bastan beri uyguladigi "AI tarafini insan
ortalamasina hizala" prensibiyle tutarli (confound onleme, bkz. proje notlari).

Cikti: data/raw/ai_humanized_qwen_gpt/ai_humanized_qwen_gpt.jsonl (YENI, kanonik
ai_humanized/ dosyasina DOKUNMAZ -- once kucuk pilotla kalite kontrolu yapilmali, sonra
elle/ayri bir adimda mevcut Qwen-backtranslate kayitlarinin YERINE gecirilmeli).

Checkpoint/resume: humanize_short_openai.py ile ayni mantik.

Kullanim:
    python scripts/humanize_qwen_main_gpt.py --target-count 3   # once pilot
    python scripts/humanize_qwen_main_gpt.py                    # sonra tam 2810
"""
from __future__ import annotations

import argparse

from humanai_detect.config import PROJECT_ROOT, get_api_key, load_yaml
from humanai_detect.data_collection import humanizers
from humanai_detect.data_collection.schemas import RawSample
from humanai_detect.utils.io import read_jsonl

SOURCE_LABEL = "ai_raw"
TARGET_LABEL = "ai_humanized_qwen_gpt"

# insan ana-havuz ortalamasina hizalanmis (bkz. docstring) -- Qwen'in kendi (ort~807) uzunlugu
# DEGIL, projenin confound-onleme prensibiyle tutarli hedef.
TARGET_LEN_MEAN = 650
TARGET_LEN_STD = 250
TARGET_LEN_MIN = 140
TARGET_LEN_MAX = 850


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-count", type=int, default=None,
                         help="Kac ornegi humanize edecegini sinirlar (test/pilot icin)")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    data_sources_cfg = load_yaml("data_sources")

    src_path = PROJECT_ROOT / paths_cfg["raw_dir"] / SOURCE_LABEL / f"{SOURCE_LABEL}.jsonl"
    all_samples = [RawSample(**r) for r in read_jsonl(src_path)]
    qwen_samples = [s for s in all_samples if "_transformers_" in s.id and "_short" not in s.id]
    print(f"[humanize-qwen-main-gpt] kaynak: {len(qwen_samples)} Qwen ana-havuz ai_raw ornegi bulundu.")

    if args.target_count is not None:
        qwen_samples = qwen_samples[: args.target_count]

    out_path = PROJECT_ROOT / paths_cfg["raw_dir"] / TARGET_LABEL / f"{TARGET_LABEL}.jsonl"

    provider_cfg = data_sources_cfg["llm_generators"]["openai"]
    rate_limit = {
        "requests_per_minute": data_sources_cfg.get("cross_generator_topup", {}).get("requests_per_minute", 8),
        "max_retries": data_sources_cfg.get("cross_generator_topup", {}).get("max_retries", 5),
    }

    existing: list[RawSample] = []
    if out_path.exists():
        existing = [RawSample(**r) for r in read_jsonl(out_path)]
        if len(existing) >= len(qwen_samples):
            print(f"[humanize-qwen-main-gpt] {len(existing)} ornek zaten mevcut, atlanıyor.")
            return
        print(f"[humanize-qwen-main-gpt] {len(existing)} mevcut ornek yuklendi, "
              f"{len(qwen_samples) - len(existing)} daha uretilecek.")

    new_samples = humanizers.humanize_batch_llm(
        qwen_samples,
        "openai",
        rate_limit=rate_limit,
        checkpoint_path=out_path,
        start_index=len(existing),
        max_concurrency=data_sources_cfg.get("cross_generator_topup", {}).get("max_concurrency", 3),
        model=provider_cfg["model"],
        api_key=get_api_key(provider_cfg["api_key_env"]),
        mean=TARGET_LEN_MEAN,
        std=TARGET_LEN_STD,
        min_words=TARGET_LEN_MIN,
        max_words=TARGET_LEN_MAX,
    )
    print(f"[humanize-qwen-main-gpt] Tamamlandi: {len(existing) + len(new_samples)} ornek.")


if __name__ == "__main__":
    main()
