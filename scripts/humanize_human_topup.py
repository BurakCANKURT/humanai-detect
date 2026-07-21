"""Insan metnini de back-translation'dan gecirip "human" etiketiyle (YENI SINIF ACMADAN)
egitime augmentasyon olarak ekler.

Amac (bkz. proje notlari, 2026-07-21 -- literatur taramasinda bulunan DAMAGE makalesi,
Pangram Labs, arXiv:2501.03437): Su ana kadar SADECE ai_raw metni back-translation ile
humanize ediliyordu. DAMAGE makalesi hem insan hem AI belgesini AYNI humanizer'dan gecirip
ikisini de egitime katiyor -- gerekce: "humanization'a modelin cevabini bir invariance
olarak ogret, ayri bir sinif degil" (modelin 'bu sadece ceviri artigi, AI imzasi degil'
ayrimini ogrenmesi icin). Ablation'larinda bu yaklasim TPR'i %96.83 -> %98.26 iyilestirdi.

KRITIK -- etiket DEGISMEZ: humanize_batch_llm'e output_label="human" verilir, ciktinin
label alani "ai_humanized" DEGIL "human" olur -- 4. bir sinif ACMIYORUZ, sadece human
sinifini paraphrase-invariance ile zenginlestiriyoruz.

Kaynak: data/raw/human/human.jsonl DEGIL, data/interim/human/human.jsonl (3262 ornek,
zaten preprocess/kalite filtresinden gecmis, guncel production'in kullandigi TAM KUME --
raw dosyada eski/kullanilmayan fazladan kayitlar var, bkz. proje notlari).

Cikti: data/raw/human_backtranslated/human_backtranslated.jsonl -- mevcut
data/raw/human/human.jsonl'e DOKUNMAZ, ayri dosya. Preprocess'ten sonra (TOPUP_LABELS,
ana human 30-850 kriteriyle AYNI) kanonik data/interim/human/human.jsonl'e birlestirilecek.

ONEMLI -- gruplama: build_features.py::_build_groups() bu script'le AYNI oturumda
guncellendi -- humanized-human ornekleri metadata.original_id uzerinden kaynak insan
belgesinin GRUBUNU miras alir (aksi halde bir belgenin orijinali ve ceviri-ikizi
train/holdout arasina bolunup dogrudan veri sizintisina yol acardi, ai_raw/ai_humanized
icin zaten uygulanan mantikla AYNI).

Checkpoint/resume: ana collect_data.py ile ayni mantik.

Kullanim:
    python scripts/humanize_human_topup.py --target-count 20   # ONCE kucuk pilot
    python scripts/humanize_human_topup.py                     # sonra TUM 3262 (varsayilan)
"""
from __future__ import annotations

import argparse

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.data_collection import humanizers
from humanai_detect.data_collection.schemas import RawSample
from humanai_detect.utils.io import read_jsonl


def _load_human_interim_as_raw(paths_cfg: dict) -> list[RawSample]:
    interim_path = PROJECT_ROOT / paths_cfg["interim_dir"] / "human" / "human.jsonl"
    if not interim_path.exists():
        raise FileNotFoundError(f"{interim_path} bulunamadi. Once ana human verisi islenmis olmali.")
    samples = []
    for rec in read_jsonl(interim_path):
        samples.append(RawSample(
            id=rec["id"],
            text=rec["text"],
            label="human",
            source=rec.get("source", "unknown"),
            metadata=rec.get("metadata") or {},
        ))
    return samples


def humanize_human(paths_cfg: dict, data_sources_cfg: dict, target_count: int | None = None) -> list[RawSample]:
    human_samples = _load_human_interim_as_raw(paths_cfg)
    if target_count is not None:
        human_samples = human_samples[:target_count]

    bt_cfg = data_sources_cfg["humanizers"]["llm"]["backtranslate"]
    out_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "human_backtranslated" / "human_backtranslated.jsonl"

    existing: list[RawSample] = []
    if out_path.exists():
        existing = [RawSample(**r) for r in read_jsonl(out_path)]
        if len(existing) >= len(human_samples):
            print(f"[human-backtranslate] {len(existing)} ornek zaten mevcut, atlanıyor.")
            return existing[: len(human_samples)]
        print(f"[human-backtranslate] {len(existing)} mevcut ornek yuklendi, {len(human_samples) - len(existing)} daha uretilecek.")

    new_samples = humanizers.humanize_batch_llm(
        human_samples,
        "backtranslate",
        checkpoint_path=out_path,
        start_index=len(existing),
        tr_en_model=bt_cfg["tr_en_model"],
        en_tr_model=bt_cfg["en_tr_model"],
        device=bt_cfg.get("device", "auto"),
        output_label="human",
        id_prefix="human_backtranslate",
    )
    return existing + new_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-count", type=int, default=None,
                         help="Kac human ornegini back-translate edecegini sinirlar (test/pilot icin)")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    data_sources_cfg = load_yaml("data_sources")

    samples = humanize_human(paths_cfg, data_sources_cfg, target_count=args.target_count)
    print(f"[human-backtranslate] Tamamlandi: {len(samples)} ornek.")


if __name__ == "__main__":
    main()
