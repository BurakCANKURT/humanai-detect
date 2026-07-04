"""DergiPark'tan otomatik Turkce (2019+) makale toplama -> data/raw/human/human.jsonl.

Kaynak: DergiPark OAI-PMH (bkz. data_collection/dergipark_harvest.py docstring'i) --
CAPTCHA/ToS kisitlamasi olmayan tek otomatiklestirilebilir insan-metni kaynagi.
Checkpoint: her kabul edilen ornek aninda dosyaya yazilir; script kesintiye ugrarsa
ayni komutla tekrar calistirildiginda data/raw/human/.dergipark_state.json sayesinde
kaldigi yerden devam eder.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.data_collection import dergipark_harvest
from humanai_detect.utils.io import read_jsonl

_JOURNAL_URL_RE = re.compile(r"/pub/([^/]+)/article/")


def _journal_of(record: dict) -> str | None:
    """Eski kayitlarda 'journal' metadata alani olmayabilir; article_url'den turetilir."""
    journal = record["metadata"].get("journal")
    if journal:
        return journal
    m = _JOURNAL_URL_RE.search(record["metadata"].get("article_url", ""))
    return m.group(1) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-total", type=int, default=3000,
        help="data/raw/human/human.jsonl icin hedeflenen toplam satir sayisi",
    )
    parser.add_argument(
        "--max-per-journal", type=int, default=5,
        help="Tek bir dergiden en fazla kac makale kabul edilecek (cesitlilik icin)",
    )
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    preprocessing_cfg = load_yaml("preprocessing")
    out_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "human" / "human.jsonl"
    state_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "human" / ".dergipark_state.json"

    existing = read_jsonl(out_path) if out_path.exists() else []
    current_total = len(existing)
    accepted_article_ids = {
        r["metadata"]["oai_id"] for r in existing
        if r.get("source") == "dergipark" and "oai_id" in r.get("metadata", {})
    }
    # Satir SAYISI degil, en yuksek mevcut ID numarasi + 1 kullanilir: aradan kayit
    # silinmis olsa bile (orn. kalite temizligi) yeni ID'ler mevcutlarla catismaz.
    existing_dp_numbers = [
        int(r["id"].rsplit("_", 1)[1])
        for r in existing
        if r.get("source") == "dergipark" and r["id"].rsplit("_", 1)[-1].isdigit()
    ]
    next_index = max(existing_dp_numbers) + 1 if existing_dp_numbers else 0

    # journal_counts MAKALE bazinda tutulur (chunk degil) -- harvest() cap'i de
    # makale bazinda uyguluyor, bu yuzden mevcut kayitlar (journal, oai_id) ikilisine
    # gore tekillestirilir.
    seen_articles: set[tuple[str, str]] = set()
    journal_counts: dict[str, int] = {}
    for r in existing:
        if r.get("source") == "dergipark":
            journal = _journal_of(r)
            oai_id = r["metadata"].get("oai_id")
            if journal and oai_id and (journal, oai_id) not in seen_articles:
                seen_articles.add((journal, oai_id))
                journal_counts[journal] = journal_counts.get(journal, 0) + 1
    if journal_counts:
        print(f"[dergipark] mevcut dergi dagilimi (makale): {journal_counts}")

    target_new = args.target_total - current_total
    if target_new <= 0:
        print(f"[dergipark] mevcut {current_total} kayit hedefe ({args.target_total}) zaten ulasmis.")
        return
    print(f"[dergipark] mevcut toplam: {current_total}, hedef: {args.target_total}, toplanacak: {target_new}")

    counter = {"n": next_index}

    def on_chunk(sample) -> None:
        sample.id = f"human_dergipark_{counter['n']:04d}"
        counter["n"] += 1
        with out_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
        print(f"  [{counter['n']}] {sample.id} yazildi.")

    stats = dergipark_harvest.harvest(
        accepted_article_ids=accepted_article_ids,
        target_new_chunks=target_new,
        min_tokens=preprocessing_cfg["min_tokens"],
        max_tokens=preprocessing_cfg["max_tokens"],
        state_path=state_path,
        on_chunk=on_chunk,
        journal_counts=journal_counts,
        max_per_journal=args.max_per_journal,
    )
    print(f"[dergipark] tamamlandi: {stats}")


if __name__ == "__main__":
    main()
