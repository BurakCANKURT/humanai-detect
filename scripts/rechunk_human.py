"""Mevcut ham insan metnini (data/raw/human/human.jsonl, ~2000 kelimelik chunk'lar) AI
uretiminin ulasabildigi uzunluga (configs/preprocessing.yaml -> max_tokens) gore yeniden
boler. PDF/kaynak yeniden indirilmez; sadece daha once cikarilmis 'text' alani tekrar
cumle sinirlarinda dilimlenir.

Girdi : data/raw/human/human.jsonl
Cikti : data/raw/human/human.jsonl (yeni chunk'lar), eskisi human_2000chunk_backup.jsonl'e tasinir
"""
from __future__ import annotations

from dataclasses import asdict

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.data_collection.file_ingest import chunk_text
from humanai_detect.data_collection.schemas import RawSample
from humanai_detect.utils.io import read_jsonl, write_jsonl


def main() -> None:
    paths_cfg = load_yaml("paths")
    prep_cfg = load_yaml("preprocessing")
    min_tokens = prep_cfg["min_tokens"]
    max_tokens = prep_cfg["max_tokens"]

    src_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "human" / "human.jsonl"
    backup_path = src_path.parent / "human_2000chunk_backup.jsonl"

    original = [RawSample(**r) for r in read_jsonl(src_path)]
    print(f"[rechunk] {len(original)} orijinal kayit okundu (max_tokens=2000 ile chunklanmisti).")

    new_samples: list[RawSample] = []
    counter = 0
    for sample in original:
        pieces = chunk_text(sample.text, min_words=min_tokens, max_words=max_tokens)
        for piece in pieces:
            metadata = dict(sample.metadata or {})
            metadata["rechunked_from"] = sample.id
            new_samples.append(
                RawSample(
                    id=f"human_rechunked_{counter:04d}",
                    text=piece,
                    label="human",
                    source=sample.source,
                    metadata=metadata,
                )
            )
            counter += 1

    print(f"[rechunk] {len(new_samples)} yeni kayit uretildi (max_tokens={max_tokens}).")

    if not backup_path.exists():
        write_jsonl(backup_path, (asdict(s) for s in original))
        print(f"[rechunk] orijinal yedeklendi -> {backup_path}")
    else:
        print(f"[rechunk] yedek zaten mevcut, ustune yazilmadi -> {backup_path}")

    write_jsonl(src_path, (asdict(s) for s in new_samples))
    print(f"[rechunk] yeni chunk'lar yazildi -> {src_path}")


if __name__ == "__main__":
    main()
