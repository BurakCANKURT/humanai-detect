"""Cok-ureticili genelleme takviyesi (bkz. scripts/collect_ai_raw_topup.py): GPT-4o-mini
ile uretilen 1000 yeni ai_raw ornegini mevcut back-translation humanizer'iyla (Helsinki-NLP
OPUS-MT, TR->EN->TR) ai_humanized'e cevirir.

Ana ai_raw/ai_humanized cift havuzuna DOKUNMAZ -- ayri data/raw/ai_raw_openai/ icin ayri bir
data/raw/ai_humanized_openai/ai_humanized_openai.jsonl uretir. Preprocess/build_features
asamasinda kisa-pilot verisiyle ayni yontemle ana havuza katilacak.

Ekstra API cagrisi GEREKMEZ (backtranslate yerel/ucretsiz bir model kullanir, ayni ai_raw
uretiminde kullanilan GPT-4o-mini'den tamamen farkli, kucuk bir seq2seq ceviri modeli).

Checkpoint/resume: ana collect_data.py ile ayni mantik.

Kullanim:
    python scripts/humanize_ai_raw_topup.py
    python scripts/humanize_ai_raw_topup.py --target-count 10  # kucuk test icin
"""
from __future__ import annotations

import argparse

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.data_collection import humanizers
from humanai_detect.data_collection.schemas import RawSample
from humanai_detect.utils.io import read_jsonl


def humanize_topup(paths_cfg: dict, data_sources_cfg: dict, target_count: int | None = None) -> list[RawSample]:
    topup_cfg = data_sources_cfg["cross_generator_topup"]
    provider = topup_cfg["provider"]  # "openai" -- kaynak ai_raw havuzunun etiketi

    src_path = PROJECT_ROOT / paths_cfg["raw_dir"] / f"ai_raw_{provider}" / f"ai_raw_{provider}.jsonl"
    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} bulunamadi. Once collect_ai_raw_topup.py calistirin.")
    ai_raw_samples = [RawSample(**r) for r in read_jsonl(src_path)]
    if target_count is not None:
        ai_raw_samples = ai_raw_samples[:target_count]

    bt_cfg = data_sources_cfg["humanizers"]["llm"]["backtranslate"]
    out_path = PROJECT_ROOT / paths_cfg["raw_dir"] / f"ai_humanized_{provider}" / f"ai_humanized_{provider}.jsonl"

    existing: list[RawSample] = []
    if out_path.exists():
        existing = [RawSample(**r) for r in read_jsonl(out_path)]
        if len(existing) >= len(ai_raw_samples):
            print(f"[humanize-topup] {len(existing)} ornek zaten mevcut, atlanıyor.")
            return existing[: len(ai_raw_samples)]
        print(f"[humanize-topup] {len(existing)} mevcut ornek yuklendi, {len(ai_raw_samples) - len(existing)} daha uretilecek.")

    new_samples = humanizers.humanize_batch_llm(
        ai_raw_samples,
        "backtranslate",
        checkpoint_path=out_path,
        start_index=len(existing),
        tr_en_model=bt_cfg["tr_en_model"],
        en_tr_model=bt_cfg["en_tr_model"],
        device=bt_cfg.get("device", "auto"),
    )
    return existing + new_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-count", type=int, default=None,
                         help="Kac ai_raw_openai ornegini humanize edecegini sinirlar (test icin)")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    data_sources_cfg = load_yaml("data_sources")

    samples = humanize_topup(paths_cfg, data_sources_cfg, target_count=args.target_count)
    print(f"[humanize-topup] Tamamlandi: {len(samples)} ornek.")


if __name__ == "__main__":
    main()
