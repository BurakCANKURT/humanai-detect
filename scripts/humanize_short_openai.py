"""Kisa-metin (5-30 kelime) havuzunu GPT-4o-mini PROMPT-TABANLI yeniden yazimla humanize eder
(back-translation DEGIL).

Amac (bkz. proje notlari, 2026-07-21): Qwen-kisa ai_raw/ai_humanized ciftlerinde olculdu ki
back-translation kisa metinde neredeyse hic degisiklik yapmiyor (char-benzerlik ort=0.854,
ana havuzun 0.17-0.26'sinin cok uzerinde) -- round-trip ceviri kisa/tek cumlelik metinde
neredeyse tersine-cevrilebilir oluyor, uzun metindeki "surukleme" birikmiyor. Bu, kisa-metinde
ai_raw/ai_humanized ayirt edilememesinin (onceden bulunan ama kok nedeni arastirilmamis bir
sorun) somut aciklamasi.

Cozum: kisa-metin havuzunda back-translation YERINE GPT-4o-mini'nin _HUMANIZE_PROMPT_TEMPLATE
ile prompt-tabanli yeniden yazimi kullanilir (humanize_with_openai, target_len short_pilot'un
5-30 kelime araligina KALIBRE EDILMIS).

Iki kaynak modu:
  --source ai_raw_short         : mevcut Qwen-kisa ai_raw (300 ornek) icin ai_humanized_short'u
                                   YENIDEN uretir. Eski (back-translation-tabanli) ai_humanized_short
                                   dosyasi SILINMEZ, data/raw/_backup_before_gpt_rehumanize/'a tasinir.
  --source ai_raw_openai_short  : yeni GPT-kisa ai_raw (collect_ai_raw_openai_short.py ciktisi)
                                   icin ai_humanized_openai_short'u ILK KEZ uretir (backup gerekmez).

Checkpoint/resume: ana collect_data.py ile ayni mantik.

Kullanim:
    python scripts/humanize_short_openai.py --source ai_raw_short --target-count 5   # once pilot
    python scripts/humanize_short_openai.py --source ai_raw_short                    # sonra tam 300
    python scripts/humanize_short_openai.py --source ai_raw_openai_short             # yeni GPT-kisa havuzu
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone

from humanai_detect.config import PROJECT_ROOT, get_api_key, load_yaml
from humanai_detect.data_collection import humanizers
from humanai_detect.data_collection.schemas import RawSample
from humanai_detect.utils.io import read_jsonl

SOURCE_TO_TARGET = {
    "ai_raw_short": "ai_humanized_short",
    "ai_raw_openai_short": "ai_humanized_openai_short",
}


def _backup_if_needed(out_path, source: str) -> None:
    """source == ai_raw_short icin mevcut back-translation-tabanli ai_humanized_short'u yedekler."""
    if source != "ai_raw_short" or not out_path.exists():
        return
    backup_dir = PROJECT_ROOT / "data" / "raw" / "_backup_before_gpt_rehumanize"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"ai_humanized_short_BACKTRANSLATE_{ts}.jsonl"
    shutil.move(str(out_path), str(backup_path))
    print(f"[humanize-short-gpt] eski (back-translation) dosya yedeklendi -> {backup_path}")


def humanize_short_gpt(paths_cfg: dict, data_sources_cfg: dict, source: str, target_count: int | None = None) -> list[RawSample]:
    src_path = PROJECT_ROOT / paths_cfg["raw_dir"] / source / f"{source}.jsonl"
    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} bulunamadi.")
    ai_raw_samples = [RawSample(**r) for r in read_jsonl(src_path)]
    if target_count is not None:
        ai_raw_samples = ai_raw_samples[:target_count]

    target_label = SOURCE_TO_TARGET[source]
    out_path = PROJECT_ROOT / paths_cfg["raw_dir"] / target_label / f"{target_label}.jsonl"
    _backup_if_needed(out_path, source)

    sp_cfg = data_sources_cfg["short_pilot"]
    provider_cfg = data_sources_cfg["llm_generators"]["openai"]
    rate_limit = {
        "requests_per_minute": data_sources_cfg.get("cross_generator_topup_short", {}).get("requests_per_minute", 8),
        "max_retries": data_sources_cfg.get("cross_generator_topup_short", {}).get("max_retries", 5),
    }

    existing: list[RawSample] = []
    if out_path.exists():
        existing = [RawSample(**r) for r in read_jsonl(out_path)]
        if len(existing) >= len(ai_raw_samples):
            print(f"[humanize-short-gpt] {len(existing)} ornek zaten mevcut, atlanıyor.")
            return existing[: len(ai_raw_samples)]
        print(f"[humanize-short-gpt] {len(existing)} mevcut ornek yuklendi, {len(ai_raw_samples) - len(existing)} daha uretilecek.")

    new_samples = humanizers.humanize_batch_llm(
        ai_raw_samples,
        "openai",
        rate_limit=rate_limit,
        checkpoint_path=out_path,
        start_index=len(existing),
        model=provider_cfg["model"],
        api_key=get_api_key(provider_cfg["api_key_env"]),
        mean=sp_cfg["target_len_mean"],
        std=sp_cfg["target_len_std"],
        min_words=sp_cfg["target_len_min"],
        max_words=sp_cfg["target_len_max"],
    )
    return existing + new_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=list(SOURCE_TO_TARGET), required=True)
    parser.add_argument("--target-count", type=int, default=None,
                         help="Kac ornegi humanize edecegini sinirlar (test/pilot icin)")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    data_sources_cfg = load_yaml("data_sources")

    samples = humanize_short_gpt(paths_cfg, data_sources_cfg, args.source, target_count=args.target_count)
    print(f"[humanize-short-gpt] Tamamlandi: {len(samples)} ornek.")


if __name__ == "__main__":
    main()
