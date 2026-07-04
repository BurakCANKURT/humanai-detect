"""Asama 1 sonrasi kalite kontrolu: data/raw/human/human.jsonl icindeki Turkce
olmayan (veya karisik dilli) kayitlari tespit edip ana dosyadan cikarir.

Manuel indirilen kaynaklarda (google_scholar, yok_tez, manual_corpus) dil kontrolu
hic yapilmiyordu; bu script py3langid ile her kaydin metnini siniflandirir. Turkce
olmayanlar silinmez, data/raw/human/human_rejected_lang.jsonl'a tasinir (geri
donduruleblir).
"""

from __future__ import annotations

import json
from pathlib import Path

import py3langid as langid

from humanai_detect.config import PROJECT_ROOT, load_yaml


def main() -> None:
    paths_cfg = load_yaml("paths")
    human_dir = PROJECT_ROOT / paths_cfg["raw_dir"] / "human"
    in_path = human_dir / "human.jsonl"
    rejected_path = human_dir / "human_rejected_lang.jsonl"

    records = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    kept, rejected = [], []
    for r in records:
        lang, _ = langid.classify(r["text"])
        if lang == "tr":
            kept.append(r)
        else:
            r["metadata"] = {**r.get("metadata", {}), "detected_lang": lang}
            rejected.append(r)

    print(f"Toplam: {len(records)} | Turkce: {len(kept)} | Turkce degil: {len(rejected)}")
    if rejected:
        from collections import Counter
        by_source = Counter(r["source"] for r in rejected)
        by_lang = Counter(r["metadata"]["detected_lang"] for r in rejected)
        print("Kaynaga gore reddedilen:", dict(by_source))
        print("Tespit edilen dile gore:", dict(by_lang))
        for r in rejected[:10]:
            print(f"  - {r['id']} ({r['metadata']['detected_lang']}): {r['text'][:80]!r}")

    with in_path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if rejected:
        mode = "a" if rejected_path.exists() else "w"
        with rejected_path.open(mode, encoding="utf-8") as f:
            for r in rejected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Reddedilenler -> {rejected_path}")


if __name__ == "__main__":
    main()
