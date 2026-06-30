"""JSONL ve Parquet okuma/yazma yardimcilari."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL dosyasini liste of dict olarak okur."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Liste of dict'i JSONL dosyasina yazar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_parquet(path: Path):
    """Parquet dosyasini pandas DataFrame olarak okur."""
    import pandas as pd

    return pd.read_parquet(path)


def write_parquet(df, path: Path) -> None:
    """pandas DataFrame'i Parquet dosyasina yazar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
