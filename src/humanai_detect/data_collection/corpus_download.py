"""Acik lisansli, hazir Turkce korpuslardan (Hugging Face datasets) insan-yazimi metin indirme.

DergiPark/YOK Tez Merkezi arama sayfalari CAPTCHA korumali oldugu icin otomatik kazima
yapilamiyor (bkz. human_sources.py). Bunun yerine manual_corpus icin, scraping riski
tasimayan, acik lisansli (CC BY-SA) Turkce Vikipedi dump'i Hugging Face uzerinden indirilir;
her madde data/external/manual_corpus/ altina ayri bir .txt dosyasi olarak yazilir ve mevcut
insan-yazimi ingestion akisina (human_sources.collect_from_manual_corpus) degisiklik
yapilmadan dahil olur.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

_FILENAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _safe_filename(title: str, idx: int) -> str:
    slug = _FILENAME_SAFE_RE.sub("_", title).strip("_")[:60]
    return f"wikipedia_tr_{idx:05d}_{slug}.txt" if slug else f"wikipedia_tr_{idx:05d}.txt"


def download_turkish_wikipedia(
    output_dir: Path,
    target_count: int,
    min_words: int = 30,
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.tr",
    dataset: Iterable[dict] | None = None,
) -> int:
    """Turkce Vikipedi maddelerini output_dir'e .txt olarak yazar, yazilan dosya sayisini dondurur.

    dataset, gercek HuggingFace yuklemesini atlayip test icin sahte madde listesi
    enjekte etmek amaciyla verilebilir; normalde None birakilir.
    """
    if dataset is None:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for idx, article in enumerate(dataset):
        if written >= target_count:
            break
        text = (article.get("text") or "").strip()
        if len(text.split()) < min_words:
            continue
        path = output_dir / _safe_filename(article.get("title", ""), idx)
        path.write_text(text, encoding="utf-8")
        written += 1
    return written
