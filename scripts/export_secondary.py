"""Asama 8: ikincil ciktilar (LLM-distance score, anomaly heatmap, vb.).

Girdi :
  --text-file   : tek bir .txt metin dosyasi
  --model       : egitilmis .pkl siniflandirici
  --reference   : data/processed/stylometric.parquet (insan referans istatistikleri)

Cikti : outputs/reports/secondary_scores/<sample_id>.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.preprocessing import clean_text
from humanai_detect.preprocessing.linguistic import analyze
from humanai_detect.preprocessing.burstiness import compute_burstiness
from humanai_detect.preprocessing.perplexity import compute_perplexity
from humanai_detect.preprocessing.schemas import ProcessedSample
from humanai_detect.features.aggregator import extract_all_features
from humanai_detect.secondary_outputs.scores import compile_secondary_scores

LABEL_NAMES = ["human", "ai_raw", "ai_humanized"]


def _build_human_reference(sty_df: pd.DataFrame) -> dict:
    human = sty_df[sty_df["label"] == "human"].drop(columns=["sample_id", "label"])
    return {
        "mean": human.mean().to_dict(),
        "std": human.std().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True, help="Cikti dosyasinda kullanilacak ID")
    parser.add_argument("--text-file", required=True, help="Analiz edilecek .txt dosyasi")
    parser.add_argument("--model", required=True, help="Egitilmis .pkl model")
    parser.add_argument("--reference", default=None, help="stylometric.parquet")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    features_cfg = load_yaml("features")
    preprocessing_cfg = load_yaml("preprocessing")

    text = Path(args.text_file).read_text(encoding="utf-8", errors="replace")

    # On isleme
    cleaned = clean_text(text)
    analysis = analyze(cleaned)
    sent_lengths = [len(s.split()) for s in analysis["sentences"]]
    burstiness = compute_burstiness(sent_lengths)
    perplexity = compute_perplexity(cleaned, preprocessing_cfg["perplexity_model_id"])

    sample = ProcessedSample(
        id=args.sample_id,
        text=text,
        cleaned_text=cleaned,
        label="human",  # gecici; model tarafindan belirlenir
        source="export",
        sentences=analysis["sentences"],
        tokens=analysis["tokens"],
        pos_tags=analysis["pos_tags"],
        dep_parse=analysis["dep_parse"],
        token_count=len(analysis["tokens"]),
        sentence_count=len(analysis["sentences"]),
        perplexity=perplexity,
        burstiness=burstiness,
    )

    feats = extract_all_features(sample, features_cfg, reference=None)

    # Siniflandirici tahmini
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    feat_names = sorted(feats.keys())
    X = np.array([[feats.get(n, 0.0) for n in feat_names]], dtype=np.float32)
    pred_int = int(model.predict(X).reshape(-1)[0])
    pred_label = LABEL_NAMES[pred_int]
    confidence = float(model.predict_proba(X)[0, pred_int]) if hasattr(model, "predict_proba") else 0.0

    # Referans istatistikleri
    ref_path = Path(args.reference) if args.reference else PROJECT_ROOT / paths_cfg["processed_dir"] / "stylometric.parquet"
    human_ref: dict = {}
    if ref_path.exists():
        sty_df = pd.read_parquet(ref_path)
        human_ref = _build_human_reference(sty_df)

    secondary = compile_secondary_scores(
        args.sample_id, feats, human_ref, pred_label, confidence
    )

    out_dir = PROJECT_ROOT / paths_cfg["reports_dir"] / "secondary_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.sample_id}.json"
    out_path.write_text(json.dumps(secondary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] Tahmin: {pred_label} (confidence={confidence:.3f})")
    print(f"[export] Ikincil skorlar -> {out_path}")


if __name__ == "__main__":
    main()
