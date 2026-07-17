"""Tek seferlik patch: data/interim/{label}/{label}.jsonl icindeki 'burstiness' alanini,
sentence_count < 2 olan kayitlarda eski sabit 0.0 yerine NaN'e cevirir.

Baglam: preprocessing/burstiness.py::compute_burstiness artik boyle durumlarda NaN donuyor
(tanimsiz varyansi yanlislikla "duzenli/AI-benzeri" 0.0 ile karistirmamak icin), ama bu
degisiklik zaten islenmis interim dosyalarina otomatik yansimaz -- Stanza pipeline'ini
(preprocess.py) tum veriyle yeniden calistirmak yerine (saatler surer), sadece etkilenen
alani duzeltmek yeterli: sentence_count < 2 durumunda varyans zaten tanimsiz, tekrar
hesaplamaya gerek yok, dogrudan NaN yazilabilir.
"""
from __future__ import annotations

import json
import math

from humanai_detect.config import PROJECT_ROOT
from humanai_detect.utils.io import read_jsonl

LABELS = ["human", "ai_raw", "ai_humanized"]


def main() -> None:
    interim_dir = PROJECT_ROOT / "data" / "interim"
    for label in LABELS:
        path = interim_dir / label / f"{label}.jsonl"
        if not path.exists():
            print(f"[{label}] dosya yok, atlanıyor.")
            continue

        records = read_jsonl(path)
        patched = 0
        for r in records:
            if r.get("sentence_count", 99) >= 2:
                continue
            current = r.get("burstiness")
            if isinstance(current, float) and math.isnan(current):
                continue
            r["burstiness"] = math.nan
            patched += 1

        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[{label}] {len(records)} kayit, {patched} tanesi burstiness=NaN olarak patch'lendi -> {path}")


if __name__ == "__main__":
    main()
