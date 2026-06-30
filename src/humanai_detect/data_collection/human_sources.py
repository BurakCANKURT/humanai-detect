"""Gercek akademik metinlerin (insan yazimi) toplanmasi - Google Scholar / YOK Tez Merkezi / manuel.

YOK Tez Merkezi arama sayfasi bot-dogrulama (CAPTCHA) ile korunuyor; otomatik kazima
bu korumayi asmaya calismak anlamina gelecegi icin yapilmiyor. Bunun yerine kullanici
makale/tez dosyalarini acik erisim lisanslarina uygun sekilde elle indirip
data/external/<kaynak>/ klasorune koyar, bu modul de o klasoru okuyup RawSample
listesine cevirir. Bkz. data/external/README.md.
"""

from __future__ import annotations

from pathlib import Path

from ..config import load_yaml
from .file_ingest import chunk_text, extract_text, iter_supported_files
from .schemas import RawSample


def _collect_from_directory(directory: Path, source: str, target_count: int) -> list[RawSample]:
    """Bir klasordeki desteklenen dosyalari okuyup RawSample listesine cevirir.

    Uzun dosyalar (orn. tam tez) configs/preprocessing.yaml'daki min/max_tokens
    araligina gore birden fazla ornege bolunur (bkz. file_ingest.chunk_text).
    """
    preprocessing_cfg = load_yaml("preprocessing")
    min_tokens = preprocessing_cfg["min_tokens"]
    max_tokens = preprocessing_cfg["max_tokens"]

    samples: list[RawSample] = []
    for path in iter_supported_files(directory):
        if len(samples) >= target_count:
            break
        text = extract_text(path).strip()
        if not text:
            continue
        for chunk in chunk_text(text, min_tokens, max_tokens):
            if len(samples) >= target_count:
                break
            samples.append(
                RawSample(
                    id=f"human_{source}_{len(samples):04d}",
                    text=chunk,
                    label="human",
                    source=source,
                    metadata={"filename": path.name},
                )
            )
    return samples


def collect_from_google_scholar(target_count: int, source_dir: Path) -> list[RawSample]:
    """data/external/google_scholar/ altina elle indirilen acik erisim makalelerden insan yazimi ornek toplar."""
    return _collect_from_directory(source_dir, "google_scholar", target_count)


def collect_from_yok_tez(target_count: int, source_dir: Path) -> list[RawSample]:
    """data/external/yok_tez/ altina elle indirilen acik erisim tezlerden insan yazimi ornek toplar."""
    return _collect_from_directory(source_dir, "yok_tez", target_count)


def collect_from_manual_corpus(target_count: int, source_dir: Path) -> list[RawSample]:
    """data/external/manual_corpus/ altina elle eklenen diger insan yazimi metinlerden ornek toplar."""
    return _collect_from_directory(source_dir, "manual_corpus", target_count)
