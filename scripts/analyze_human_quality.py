"""data/raw/human/human.jsonl icin kalite analizi.

Kontroller:
  1. Bozuk font/encoding (dergipark_harvest'teki Turkce durak-kelime orani)
  2. Dil guveni (py3langid skoru)
  3. Kaynakca/referans agirlikli chunk'lar (duz metin degil liste gibi gorunenler)
  4. Kelime sayisi dagilimi (preprocessing min/max_tokens ile karsilastirma)
  5. Yaklasik tekrar (ilk 200 karakter imzasi ile)
"""

from __future__ import annotations

import json
import re
from collections import Counter

import py3langid as langid

from humanai_detect.config import PROJECT_ROOT, load_yaml

_STOPWORDS = {
    "ve", "bir", "bu", "için", "ile", "de", "da", "olarak", "olan", "gibi",
    "en", "çok", "daha", "ancak", "ise", "ama", "veya", "ki", "mi", "mu",
    "ne", "her", "tüm", "sonra", "önce", "kadar", "göre", "ya", "yani",
}


def turkish_validity_ratio(text: str) -> float:
    words = re.findall(r"[a-zA-ZçÇğĞıİöÖşŞüÜ]+", text.lower())
    if len(words) < 30:
        return 0.0
    hits = sum(1 for w in words if w in _STOPWORDS)
    return hits / len(words)


def bibliography_ratio(text: str) -> float:
    """Kaynakca listesi gibi gorunen satirlarin orani (yil-parantez + coklu virgul deseni)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not lines:
        return 0.0
    biblio_pattern = re.compile(r"\(\d{4}\)|\d{4}\)\.|pp\.\s*\d|ss\.\s*\d")
    hits = sum(1 for l in lines if biblio_pattern.search(l))
    return hits / len(lines)


def main() -> None:
    paths_cfg = load_yaml("paths")
    preprocessing_cfg = load_yaml("preprocessing")
    in_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "human" / "human.jsonl"

    records = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Toplam kayit: {len(records)}\n")

    word_counts = []
    garbled = []
    low_lang_conf = []
    biblio_heavy = []
    by_source_garbled = Counter()
    by_source_total = Counter()

    seen_sig: dict[str, list[str]] = {}
    near_dupes = []

    for r in records:
        text = r["text"]
        wc = len(text.split())
        word_counts.append(wc)
        by_source_total[r["source"]] += 1

        ratio = turkish_validity_ratio(text)
        if ratio < 0.03:
            garbled.append((r["id"], r["source"], ratio))
            by_source_garbled[r["source"]] += 1

        lang, conf = langid.classify(text)
        if lang != "tr" or conf < -300:
            low_lang_conf.append((r["id"], r["source"], lang, round(float(conf), 1)))

        bratio = bibliography_ratio(text)
        if bratio > 0.35:
            biblio_heavy.append((r["id"], r["source"], round(bratio, 2)))

        sig = text[:200]
        seen_sig.setdefault(sig, []).append(r["id"])

    for sig, ids in seen_sig.items():
        if len(ids) > 1:
            near_dupes.append(ids)

    min_tok, max_tok = preprocessing_cfg["min_tokens"], preprocessing_cfg["max_tokens"]
    too_short = sum(1 for w in word_counts if w < min_tok)
    too_long = sum(1 for w in word_counts if w > max_tok * 1.5)

    print("=== KELIME SAYISI ===")
    print(f"  min={min(word_counts)} max={max(word_counts)} ortalama={sum(word_counts)/len(word_counts):.0f}")
    print(f"  preprocessing min_tokens({min_tok}) altinda: {too_short}")
    print(f"  preprocessing max_tokens({max_tok})*1.5 ustunde: {too_long}")

    print("\n=== BOZUK FONT/ENCODING (Turkce durak-kelime orani < 0.03) ===")
    print(f"  toplam: {len(garbled)} / {len(records)}")
    print(f"  kaynaga gore: {dict(by_source_garbled)} (toplam kayit: {dict(by_source_total)})")
    for gid, src, ratio in garbled[:10]:
        print(f"    - {gid} ({src}) oran={ratio:.3f}")

    print("\n=== DIL GUVENI DUSUK / TURKCE DEGIL ===")
    print(f"  toplam: {len(low_lang_conf)} / {len(records)}")
    for lid, src, lang, conf in low_lang_conf[:10]:
        print(f"    - {lid} ({src}) dil={lang} conf={conf}")

    print("\n=== KAYNAKCA-AGIRLIKLI CHUNK'LAR (duz metin degil) ===")
    print(f"  toplam: {len(biblio_heavy)} / {len(records)}")
    for bid, src, bratio in biblio_heavy[:10]:
        print(f"    - {bid} ({src}) oran={bratio}")

    print("\n=== YAKLASIK TEKRAR (ilk 200 karakter ayni) ===")
    print(f"  grup sayisi: {len(near_dupes)}")
    for ids in near_dupes[:10]:
        print(f"    - {ids}")


if __name__ == "__main__":
    main()
