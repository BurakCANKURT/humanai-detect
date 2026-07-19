"""Tek seferlik yardimci script: canli tahmin arayuzu icin gereken artefaktlari uretir.

Ana pipeline'i (src/humanai_detect, configs/, data/) HICBIR SEKILDE degistirmez;
sadece mevcut data/interim ve data/processed dosyalarini okuyup arayuz/artifacts/
altina turetilmis (kucuk) yardimci dosyalar yazar:

  artifacts/reference.json        -- build_features.py::_build_reference ile birebir ayni mantik
                                      (insan egitim kumesi word_freqs, mean_dep_depth, conjunction_density)
  artifacts/scaler.npz             -- fused.parquet uretilirken kullanilan z-score mean/std
                                      (stilometri blogu + BERTurk embedding blogu, ayri ayri)
  artifacts/feature_columns.json  -- egitimde kullanilan nihai ozellik sirasi (stacking modeli bu sirayi bekler)
  artifacts/length_residualizer.json -- data/processed/length_residualizer.json'un birebir kopyasi
                                         (build_features.py::fit_length_residualizer ciktisi; uzunluk-confound'lu
                                         4 ozelligi (kl_div_word_freq, entropy_2/3gram, burstiness) ayni
                                         katsayilarla duzeltmek icin canli cikarimda da kullanilir)

Sadece bir kez (ya da data/processed yeniden uretildiginde) calistirilmasi yeterlidir.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ARAYUZ_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARAYUZ_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from humanai_detect.config import load_yaml  # noqa: E402
from humanai_detect.preprocessing.schemas import ProcessedSample  # noqa: E402
from humanai_detect.features.syntactic import _word_depths  # noqa: E402
from humanai_detect.features.discourse import conjunction_density  # noqa: E402
from humanai_detect.utils.io import read_jsonl  # noqa: E402

ARTIFACTS_DIR = ARAYUZ_DIR / "artifacts"

# 2026-07-18 oturumunda eklenen 5 ozellik ailesine ait sutunlar -- ablation deneyi
# (bkz. scripts/evaluate_short_pilot.py) bunlarin katkisinin ana held-out sette
# olcum-gurultusu seviyesinde oldugunu gosterdi, production modeli bu ozellikler
# OLMADAN (35 stilometrik + 768 embedding = 803) egitildi. arayuz'un hesapladigi
# ozellik kumesi de zaten bunlari hic icermiyordu (inference.py guncellenmemisti);
# buradaki disaridan-birak islemi model/inference/artifact ucunun tutarli olmasini
# saglar.
NEW_FEATURES = [
    "perplexity_ratio", "mean_token_rank", "frac_rank_top1", "frac_rank_top5", "frac_rank_top10",
    "rank_entropy", "lexical_coherence", "ai_cliche_density", "human_informality_density",
    "punct_irregularity_rate", "double_space_rate", "post_punct_case_irregularity_rate",
]


def build_reference() -> dict:
    paths_cfg = load_yaml("paths")
    interim_dir = PROJECT_ROOT / paths_cfg["interim_dir"]
    human_path = interim_dir / "human" / "human.jsonl"
    print(f"[reference] {human_path} okunuyor...")
    human_samples = [ProcessedSample(**r) for r in read_jsonl(human_path)]
    print(f"[reference] {len(human_samples)} insan ornegi yuklendi.")

    all_tokens: list[str] = []
    dep_depths: list[float] = []
    conj_densities: list[float] = []
    for s in human_samples:
        all_tokens.extend(s.tokens)
        if s.dep_parse:
            dep_depths.extend(_word_depths(s.dep_parse))
        conj_densities.append(conjunction_density(s.tokens, s.pos_tags))

    total = len(all_tokens)
    counts = Counter(all_tokens)
    reference = {
        "word_freqs": {w: c / total for w, c in counts.items()},
        "mean_dep_depth": statistics.mean(dep_depths) if dep_depths else 0.0,
        "conjunction_density": statistics.mean(conj_densities) if conj_densities else 0.0,
    }
    print(f"[reference] {len(reference['word_freqs'])} benzersiz kelime, "
          f"mean_dep_depth={reference['mean_dep_depth']:.3f}, "
          f"conjunction_density={reference['conjunction_density']:.3f}")
    return reference


def build_scaler() -> tuple[dict, list[str]]:
    paths_cfg = load_yaml("paths")
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]

    sty_df = pd.read_parquet(processed_dir / "stylometric.parquet")
    emb_df = pd.read_parquet(processed_dir / "embeddings_berturk.parquet")

    meta_cols = {"sample_id", "label"}
    sty_cols = [c for c in sty_df.columns if c not in meta_cols and c not in NEW_FEATURES]
    emb_cols = [c for c in emb_df.columns if c not in meta_cols]

    sty_arr = sty_df[sty_cols].to_numpy(dtype=np.float32)
    emb_arr = emb_df[emb_cols].to_numpy(dtype=np.float32)

    # fuse() ile birebir ayni: zscore, train_mask=None (fused.parquet tum veriden uretildi)
    sty_mean = np.nanmean(sty_arr, axis=0)
    sty_std = np.nanstd(sty_arr, axis=0)
    sty_std = np.where(sty_std == 0, 1.0, sty_std)

    emb_mean = emb_arr.mean(axis=0)
    emb_std = emb_arr.std(axis=0)
    emb_std = np.where(emb_std == 0, 1.0, emb_std)

    scaler = {
        "sty_cols": sty_cols,
        "sty_mean": sty_mean,
        "sty_std": sty_std,
        "emb_mean": emb_mean,
        "emb_std": emb_std,
    }
    feature_columns = sty_cols + [f"emb_berturk_{c}" for c in emb_cols]
    print(f"[scaler] stilometri: {len(sty_cols)} ozellik, embedding: {len(emb_cols)} boyut, "
          f"toplam: {len(feature_columns)} ozellik")

    # fused_ablation.parquet ile karsilastirma (production modeli bu dosyayla egitildi)
    fused_path = processed_dir / "fused_ablation.parquet"
    if fused_path.exists():
        fused_cols = [c for c in pd.read_parquet(fused_path).columns if c not in meta_cols]
        if fused_cols == feature_columns:
            print("[scaler] fused_ablation.parquet sutun sirasiyla birebir uyusuyor.")
        else:
            print("[scaler] UYARI: fused_ablation.parquet sutun sirasi farkli! "
                  f"(fused={len(fused_cols)} vs turetilen={len(feature_columns)})")

    return scaler, feature_columns


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    reference = build_reference()
    (ARTIFACTS_DIR / "reference.json").write_text(
        json.dumps(reference, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[build_artifacts] -> {ARTIFACTS_DIR / 'reference.json'}")

    scaler, feature_columns = build_scaler()
    np.savez(
        ARTIFACTS_DIR / "scaler.npz",
        sty_cols=np.array(scaler["sty_cols"]),
        sty_mean=scaler["sty_mean"],
        sty_std=scaler["sty_std"],
        emb_mean=scaler["emb_mean"],
        emb_std=scaler["emb_std"],
    )
    print(f"[build_artifacts] -> {ARTIFACTS_DIR / 'scaler.npz'}")

    (ARTIFACTS_DIR / "feature_columns.json").write_text(
        json.dumps(feature_columns, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[build_artifacts] -> {ARTIFACTS_DIR / 'feature_columns.json'}")

    paths_cfg = load_yaml("paths")
    length_params_path = PROJECT_ROOT / paths_cfg["processed_dir"] / "length_residualizer.json"
    if length_params_path.exists():
        (ARTIFACTS_DIR / "length_residualizer.json").write_text(
            length_params_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(f"[build_artifacts] -> {ARTIFACTS_DIR / 'length_residualizer.json'}")
    else:
        print(f"[build_artifacts] UYARI: {length_params_path} yok, uzunluk-residualizasyonu arayuzde uygulanmayacak.")


if __name__ == "__main__":
    main()
