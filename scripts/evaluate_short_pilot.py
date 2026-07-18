"""Kisa-pilot (5-30 kelime, grup-ayrik) verisini egitilmis modelle degerlendirir.

Amac: Bu oturumda eklenen 5 yeni ozellik ailesinin, daha once olculen kisa-metin
ai_raw<->ai_humanized karisikligini (%83, bkz. proje notlari 2026-07-17) azaltip
azaltmadigini olcmek -- bu projenin asil basari kriteri.

Kisa-pilot ornekleri (data/interim/{human_short,ai_raw_short,ai_humanized_short})
ANA egitim havuzuna DAHIL DEGIL (model bunlari hic gormedi -- gercek OOD testi).
Bu yuzden feature cikarimi ANA egitimde kullanilan referans/lexicon/length-residualizer/
scaler parametreleriyle (train-mask'tan fit edilmis, data/processed/*) yapilir --
kisa-pilot verisinden YENIDEN fit edilmez (build_features.py'nin reference/lexicon/
length-residualizer mantigiyla ayni, ama kisa-pilot ozel bir "apply-only" gecisi).

Girdi : data/interim/{human_short,ai_raw_short,ai_humanized_short}/*.jsonl
        (augment_features.py ile bu oturumun yeni alanlarini ONCEDEN almis olmali)
        data/processed/{stylometric,embeddings_berturk}.parquet + holdout_ids.txt
        (train-only scaler mean/std'yi yeniden turetmek icin)
        data/processed/length_residualizer.json, data/processed/lexicons/*.json
Cikti : confusion matrix + per-class F1, konsola yazdirilir + JSON olarak kaydedilir.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.embeddings.berturk import embed_berturk
from humanai_detect.features.aggregator import extract_all_features
from humanai_detect.features.discourse import conjunction_density
from humanai_detect.features.syntactic import _word_depths
from humanai_detect.fusion.length_residualize import apply_length_residualizer_df
from humanai_detect.preprocessing.schemas import ProcessedSample
from humanai_detect.utils.io import read_jsonl

SHORT_LABELS = ["human_short", "ai_raw_short", "ai_humanized_short"]
LABEL_MAP = {"human_short": "human", "ai_raw_short": "ai_raw", "ai_humanized_short": "ai_humanized"}
LABEL_NAMES = ["human", "ai_raw", "ai_humanized"]
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(LABEL_NAMES)}


def _build_reference(interim_dir) -> dict:
    """build_features.py::_build_reference ile birebir ayni -- ANA human korpusundan."""
    human_path = interim_dir / "human" / "human.jsonl"
    human_samples = [ProcessedSample(**r) for r in read_jsonl(human_path)]

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
    return {
        "word_freqs": {w: c / total for w, c in counts.items()},
        "mean_dep_depth": statistics.mean(dep_depths) if dep_depths else 0.0,
        "conjunction_density": statistics.mean(conj_densities) if conj_densities else 0.0,
    }


def _load_lexicons(processed_dir, paths_cfg: dict) -> dict:
    """build_features.py::_load_lexicons ile birebir ayni."""
    lexicon_dir = PROJECT_ROOT / paths_cfg.get("lexicon_dir", "data/processed/lexicons")
    lexicons = {"ai_cliche": set(), "human_informality": set()}
    ai_path = lexicon_dir / "ai_cliche_ngrams.json"
    human_path = lexicon_dir / "human_informality_ngrams.json"
    if ai_path.exists():
        lexicons["ai_cliche"] = set(json.loads(ai_path.read_text(encoding="utf-8")).keys())
    if human_path.exists():
        lexicons["human_informality"] = set(json.loads(human_path.read_text(encoding="utf-8")).keys())
    return lexicons


def _fit_train_zscore(arr: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fusion/standardize.py::zscore_standardize ile ayni fit mantigi (train-only)."""
    fit_data = arr[train_mask]
    mean = np.nanmean(fit_data, axis=0)
    std = np.nanstd(fit_data, axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="_diag_after_calibrated.pkl", help="outputs/models/ altindaki model dosyasi")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    features_cfg = load_yaml("features")
    emb_cfg = load_yaml("embeddings")
    interim_dir = PROJECT_ROOT / paths_cfg["interim_dir"]
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]

    print("[short-pilot-eval] ana referans + lexicon yukleniyor...")
    reference = _build_reference(interim_dir)
    lexicons = _load_lexicons(processed_dir, paths_cfg)
    print(f"  reference: {len(reference['word_freqs'])} kelime, "
          f"lexicons: ai_cliche={len(lexicons['ai_cliche'])}, human_informality={len(lexicons['human_informality'])}")

    length_params_path = processed_dir / "length_residualizer.json"
    length_params = json.loads(length_params_path.read_text(encoding="utf-8")) if length_params_path.exists() else {}

    # --- Kisa-pilot orneklerini yukle, ozellik cikar ---
    all_samples: list[ProcessedSample] = []
    rows: list[dict] = []
    token_counts: list[int] = []
    for short_label in SHORT_LABELS:
        path = interim_dir / short_label / f"{short_label}.jsonl"
        if not path.exists():
            print(f"[short-pilot-eval] UYARI: {path} yok, atlanıyor.")
            continue
        records = list(read_jsonl(path))
        missing_new = sum(1 for r in records if "perplexity_ratio" not in r)
        if missing_new:
            print(f"[short-pilot-eval] UYARI: {short_label} icinde {missing_new} kayit yeni alanlari icermiyor "
                  "(augment_features.py --label bu-etiket ile once calistirilmali).")
        samples = [ProcessedSample(**r) for r in records]
        all_samples.extend(samples)
        print(f"[{short_label}] {len(samples)} ornek yuklendi.")
        for s in samples:
            feats = extract_all_features(s, features_cfg, reference=reference, lexicons=lexicons)
            feats["sample_id"] = s.id
            feats["label"] = LABEL_MAP[short_label]
            rows.append(feats)
            token_counts.append(s.token_count)

    if not rows:
        print("[short-pilot-eval] hicbir ornek yuklenemedi, cikiliyor.")
        return

    sty_df = pd.DataFrame(rows)
    meta_cols = {"sample_id", "label"}
    sty_cols_pilot = [c for c in sty_df.columns if c not in meta_cols]

    # --- Uzunluk-residualizasyonu (ANA train'den fit edilmis parametrelerle, apply-only) ---
    token_counts_arr = np.array(token_counts, dtype=float)
    sty_df = apply_length_residualizer_df(sty_df, token_counts_arr, length_params)

    # --- BERTurk embedding (kisa-pilot metinleri icin, lokal) ---
    print("[short-pilot-eval] BERTurk embedding hesaplaniyor...")
    berturk_cfg = emb_cfg["models"]["berturk"]
    texts = [s.cleaned_text for s in all_samples]
    emb_matrix = embed_berturk(
        texts, model_id=berturk_cfg["model_id"], pooling=emb_cfg.get("pooling", "cls"),
        max_length=emb_cfg.get("max_length", 512), batch_size=emb_cfg.get("batch_size", 16),
        device=emb_cfg.get("device", "auto"), cache_dir=None,
    )

    # --- ANA fused.parquet'in insa edildigi TREN-ONLY mean/std'yi yeniden turet ---
    print("[short-pilot-eval] ana egitim scaler'i (train-only) yeniden turetiliyor...")
    main_sty_df = pd.read_parquet(processed_dir / "stylometric.parquet")
    main_emb_df = pd.read_parquet(processed_dir / "embeddings_berturk.parquet")
    holdout_ids = set((processed_dir / "holdout_ids.txt").read_text(encoding="utf-8").splitlines())
    main_train_mask = ~main_sty_df["sample_id"].isin(holdout_ids).to_numpy()

    main_sty_cols = [c for c in main_sty_df.columns if c not in meta_cols]
    main_emb_cols = [c for c in main_emb_df.columns if c not in meta_cols]

    assert sty_cols_pilot == main_sty_cols or set(sty_cols_pilot) == set(main_sty_cols), (
        "kisa-pilot stilometrik ozellik kumesi ana egitimle uyusmuyor -- features.yaml degismis olabilir"
    )
    # Ana sutun SIRASINI kullan (fused.parquet'teki sirayla birebir ayni olmali)
    sty_df = sty_df[["sample_id", "label"] + main_sty_cols]

    sty_mean, sty_std = _fit_train_zscore(main_sty_df[main_sty_cols].to_numpy(dtype=np.float32), main_train_mask)
    emb_mean, emb_std = _fit_train_zscore(main_emb_df[main_emb_cols].to_numpy(dtype=np.float32), main_train_mask)

    pilot_sty_arr = sty_df[main_sty_cols].to_numpy(dtype=np.float32)
    pilot_sty_std = np.nan_to_num((pilot_sty_arr - sty_mean) / sty_std, nan=0.0)
    pilot_emb_std = np.nan_to_num((emb_matrix - emb_mean) / emb_std, nan=0.0)

    X_pilot = np.hstack([pilot_sty_std, pilot_emb_std]).astype(np.float32)

    # fused.parquet'in GERCEK sutun sirasiyla dogrulama
    fused_cols = [c for c in pd.read_parquet(processed_dir / "fused.parquet").columns if c not in meta_cols]
    expected_cols = main_sty_cols + [f"emb_berturk_{c}" for c in main_emb_cols]
    if fused_cols != expected_cols:
        print("[short-pilot-eval] UYARI: turetilen sutun sirasi fused.parquet ile FARKLI, "
              "sonuclar guvenilmez olabilir!")
        print(f"  beklenen (ilk 5): {expected_cols[:5]}")
        print(f"  fused.parquet (ilk 5): {fused_cols[:5]}")

    y_pilot = sty_df["label"].map(LABEL_TO_INT).to_numpy()

    # --- Model yukle, tahmin et ---
    model_path = PROJECT_ROOT / paths_cfg["models_dir"] / args.model
    print(f"[short-pilot-eval] model yukleniyor: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_pilot)

    acc = accuracy_score(y_pilot, y_pred)
    macro_f1 = f1_score(y_pilot, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_pilot, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_pilot, y_pred, labels=[0, 1, 2])

    print(f"\n[short-pilot-eval] n={len(y_pilot)}  dagilim={dict(sty_df['label'].value_counts())}")
    print(f"Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")
    for i, lbl in enumerate(LABEL_NAMES):
        print(f"  F1({lbl}) = {per_class_f1[i]:.4f}")
    print("\nConfusion matrix (satir=gercek, sutun=tahmin):")
    print("           " + "  ".join(f"{l:>13}" for l in LABEL_NAMES))
    for i, lbl in enumerate(LABEL_NAMES):
        print(f"{lbl:>10} " + "  ".join(f"{cm[i][j]:>13}" for j in range(3)))

    ai_raw_idx, ai_hum_idx = 1, 2
    total_ai_raw = cm[ai_raw_idx].sum()
    total_ai_hum = cm[ai_hum_idx].sum()
    confused_raw_as_hum = cm[ai_raw_idx][ai_hum_idx]
    confused_hum_as_raw = cm[ai_hum_idx][ai_raw_idx]
    overall_ai_confusion = (confused_raw_as_hum + confused_hum_as_raw) / (total_ai_raw + total_ai_hum) if (total_ai_raw + total_ai_hum) else 0.0

    print(f"\nai_raw -> ai_humanized: {confused_raw_as_hum}/{total_ai_raw} ({confused_raw_as_hum/total_ai_raw*100 if total_ai_raw else 0:.1f}%)")
    print(f"ai_humanized -> ai_raw: {confused_hum_as_raw}/{total_ai_hum} ({confused_hum_as_raw/total_ai_hum*100 if total_ai_hum else 0:.1f}%)")
    print(f"TOPLAM ai_raw<->ai_humanized karisikligi: {overall_ai_confusion*100:.1f}%  (onceki baseline: %83)")

    out_dir = PROJECT_ROOT / paths_cfg["reports_dir"] / "cv_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "n": int(len(y_pilot)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": {LABEL_NAMES[i]: float(per_class_f1[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABEL_NAMES,
        "ai_raw_to_ai_humanized_pct": float(confused_raw_as_hum / total_ai_raw * 100) if total_ai_raw else None,
        "ai_humanized_to_ai_raw_pct": float(confused_hum_as_raw / total_ai_hum * 100) if total_ai_hum else None,
        "overall_ai_confusion_pct": float(overall_ai_confusion * 100),
        "previous_baseline_confusion_pct": 83.0,
    }
    out_path = out_dir / "short_pilot_eval.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[short-pilot-eval] -> {out_path}")


if __name__ == "__main__":
    main()
