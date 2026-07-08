"""Asama 1 sonrasi kalite kontrolu: data/raw/ai_raw/ai_raw.jsonl icindeki Cince/
Japonca/Korece (CJK) karakter kirlenmesi olan kayitlari cikarir, ai_humanized.jsonl'deki
eslesen ciftleri de (metadata.original_id uzerinden) birlikte cikarir.

Qwen2.5-7B-Instruct (cok dilli, Cin merkezli bir model) bazi uretimlerde "dil
kaymasi" yasayip Turkce yerine kismen/tamamen CJK metin uretebiliyor (bkz. proje
notlari, 2026-07-08 -- 3000 kayittan 227'sinde tespit edildi). Cikarilan kayitlar
silinmez, *_rejected_cjk.jsonl dosyalarina tasinir (geri donduruleblir).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from humanai_detect.config import PROJECT_ROOT, load_yaml

_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-퟿]")
_CJK_RATIO_THRESHOLD = 0.01


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK_RE.findall(text)) / len(text)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    paths_cfg = load_yaml("paths")
    raw_dir = PROJECT_ROOT / paths_cfg["raw_dir"]
    ai_raw_path = raw_dir / "ai_raw" / "ai_raw.jsonl"
    ai_humanized_path = raw_dir / "ai_humanized" / "ai_humanized.jsonl"

    ai_raw_records = _read_jsonl(ai_raw_path)
    ai_humanized_records = _read_jsonl(ai_humanized_path)

    bad_ids = {r["id"] for r in ai_raw_records if _cjk_ratio(r["text"]) > _CJK_RATIO_THRESHOLD}

    kept_ai_raw = [r for r in ai_raw_records if r["id"] not in bad_ids]
    rejected_ai_raw = [r for r in ai_raw_records if r["id"] in bad_ids]

    kept_humanized = [r for r in ai_humanized_records if r["metadata"]["original_id"] not in bad_ids]
    rejected_humanized = [r for r in ai_humanized_records if r["metadata"]["original_id"] in bad_ids]

    print(f"ai_raw:       toplam={len(ai_raw_records)}  kirli(CJK>%{_CJK_RATIO_THRESHOLD*100:.0f})={len(rejected_ai_raw)}  temiz={len(kept_ai_raw)}")
    print(f"ai_humanized: toplam={len(ai_humanized_records)}  eslesip-cikarilan={len(rejected_humanized)}  temiz={len(kept_humanized)}")

    _write_jsonl(ai_raw_path, kept_ai_raw)
    _write_jsonl(ai_humanized_path, kept_humanized)
    _write_jsonl(raw_dir / "ai_raw" / "ai_raw_rejected_cjk.jsonl", rejected_ai_raw)
    _write_jsonl(raw_dir / "ai_humanized" / "ai_humanized_rejected_cjk.jsonl", rejected_humanized)

    print(f"Kirli ai_raw -> {raw_dir / 'ai_raw' / 'ai_raw_rejected_cjk.jsonl'}")
    print(f"Kirli ai_humanized -> {raw_dir / 'ai_humanized' / 'ai_humanized_rejected_cjk.jsonl'}")


if __name__ == "__main__":
    main()
