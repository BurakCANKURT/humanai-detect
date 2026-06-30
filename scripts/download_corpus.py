"""Asama 1 yardimci: manual_corpus icin acik lisansli Turkce Vikipedi'den metin indirir.

Scraping yerine Hugging Face uzerinden hazir, acik lisansli (CC BY-SA) bir Turkce
korpus indirilir; CAPTCHA/ToS riski tasimaz. collect_data.py --label human bu
dosyalari otomatik okuyup data/raw/human/human.jsonl'a ekler.

Girdi : configs/data_sources.yaml -> corpus_download + configs/preprocessing.yaml
Cikti : data/external/manual_corpus/*.txt
"""

from __future__ import annotations

import argparse

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.data_collection.corpus_download import download_turkish_wikipedia


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-count", type=int, default=None, help="Indirilecek madde sayisi (varsayilan: config)")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    data_sources_cfg = load_yaml("data_sources")
    preprocessing_cfg = load_yaml("preprocessing")
    corpus_cfg = data_sources_cfg["corpus_download"]["wikipedia_tr"]

    output_dir = PROJECT_ROOT / paths_cfg["external_dir"] / "manual_corpus"
    target_count = args.target_count or corpus_cfg["target_count"]

    written = download_turkish_wikipedia(
        output_dir=output_dir,
        target_count=target_count,
        min_words=preprocessing_cfg["min_tokens"],
        dataset_name=corpus_cfg["dataset_name"],
        dataset_config=corpus_cfg["dataset_config"],
    )
    print(f"[manual_corpus] {written} Vikipedi maddesi yazildi -> {output_dir}")


if __name__ == "__main__":
    main()
