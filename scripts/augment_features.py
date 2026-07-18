"""Mevcut data/interim/*.jsonl kayitlarina, sifirdan yeniden isleme yapmadan SADECE
yeni eklenen model-tabanli alanlari (perplexity_ratio, token-rank istatistikleri) ekler.

Amac: scripts/preprocess.py'nin tamamini (clean_text + Stanza + ilk perplexity +
burstiness) tekrar calistirmak yerine -- bunlar interim dosyalarda zaten mevcut ve bu
oturumda degismedi -- sadece 2 yeni model-tabanli ozelligi hesaplayip mevcut kayitlari
YERINDE (in-place) gunceller. ~9500 ornek icin Stanza/ilk-perplexity/burstiness'i tekrar
hesaplamamak onemli GPU suresi tasarrufu saglar.

Girdi/Cikti: data/interim/{label}/{label}.jsonl (ayni dosya, in-place guncellenir)

Checkpoint: bir kayitta "perplexity_ratio" alani ZATEN varsa (bu script tarafindan
daha once eklenmisse) atlanir -- kesinti sonrasi kaldigi yerden devam eder. Eski
kayitlarda bu alan hic yok (KeyError degil, dict'te anahtar yok), bu yuzden "in rec"
kontrolu guvenilir bir isaretleyicidir (perplexity_ratio=1.0 gibi bir varsayilan
degerle karsilastirmaktan farkli olarak yanlislikla atlama riski tasimaz).
"""

from __future__ import annotations

import argparse
import json
import math

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.preprocessing import compute_perplexity, compute_token_rank_stats
from humanai_detect.utils.io import read_jsonl

LABELS = ["human", "ai_raw", "ai_humanized"]

_NEW_FIELDS = ("perplexity_ratio", "mean_token_rank", "frac_rank_top1", "frac_rank_top5", "frac_rank_top10", "rank_entropy")


def _write_all(path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def augment_label(label: str, interim_dir, preprocessing_cfg: dict, limit: int | None = None) -> None:
    path = interim_dir / label / f"{label}.jsonl"
    if not path.exists():
        print(f"[{label}] {path} bulunamadi, atlanıyor.")
        return

    records = list(read_jsonl(path))
    total = len(records)
    already_done = sum(1 for r in records if "perplexity_ratio" in r)
    print(f"[{label}] {total} kayit yuklendi, {already_done} zaten guncellenmis (checkpoint).")

    updated = 0
    for i, rec in enumerate(records, 1):
        if limit is not None and updated >= limit:
            break
        if "perplexity_ratio" in rec:
            continue

        text = rec["cleaned_text"]
        p1 = rec["perplexity"]
        p2 = compute_perplexity(text, preprocessing_cfg["perplexity_ratio_model_id"])
        rec["perplexity_ratio"] = (
            p1 / p2 if math.isfinite(p1) and math.isfinite(p2) and p2 > 0 else 1.0
        )

        rank_stats = compute_token_rank_stats(text, preprocessing_cfg["causal_lm_model_id"])
        rec.update(rank_stats)

        # VRAM birikimini onle (preprocess.py ile ayni desen)
        try:
            import torch, gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

        updated += 1
        if i % 10 == 0 or i == total:
            print(f"  [{i}/{total}] {label} ({updated} bu oturumda guncellendi)", flush=True)

        # Her 25 guncellemede bir diske yaz (checkpoint) -- kesinti aninda ilerlemeyi kaybettirme
        if updated % 25 == 0:
            _write_all(path, records)

    _write_all(path, records)
    remaining = sum(1 for r in records if "perplexity_ratio" not in r)
    print(f"[{label}] tamamlandi -- bu oturumda {updated} kayit guncellendi, kalan: {remaining}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=[*LABELS, "all"], default="all")
    parser.add_argument("--input-dir", default=None, help="data/interim dizini (varsayilan: configs/paths.yaml)")
    parser.add_argument("--limit", type=int, default=None, help="Sinif basina bu oturumda en fazla kac kayit guncellenecek")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    preprocessing_cfg = load_yaml("preprocessing")
    interim_dir = PROJECT_ROOT / (args.input_dir or paths_cfg["interim_dir"])

    labels = LABELS if args.label == "all" else [args.label]
    for label in labels:
        augment_label(label, interim_dir, preprocessing_cfg, limit=args.limit)


if __name__ == "__main__":
    main()
