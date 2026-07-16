"""Kalibrasyon oncesi/sonrasi karsilastirma: stacking modelini once calibrate=False
(eski davranis), sonra calibrate=True (CalibratedClassifierCV, isotonic, cv=5) ile
gelistirme havuzunda egitip held-out sette Brier skoru + reliability (calibration
curve) verisiyle karsilastirir. Accuracy/macro-F1'in bozulmadigini da dogrular.

Girdi : data/processed/fused.parquet, data/processed/holdout_ids.txt, configs/models.yaml
Cikti : outputs/reports/cv_results/calibration_before_after.json/.md
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.models.train import train_final_model

LABEL_NAMES = ["human", "ai_raw", "ai_humanized"]
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(LABEL_NAMES)}


def _multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int = 3) -> dict[str, float]:
    """Her sinif icin one-vs-rest Brier skoru + ortalama."""
    per_class = {}
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        per_class[LABEL_NAMES[c]] = float(brier_score_loss(y_bin, y_proba[:, c]))
    per_class["mean"] = float(np.mean(list(per_class.values())))
    return per_class


def _reliability_bins(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int = 3, n_bins: int = 10):
    """Her sinif icin (bin_ortalama_tahmin, bin_gercek_oran) noktalarini dondurur."""
    bins = {}
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        prob_true, prob_pred = calibration_curve(y_bin, y_proba[:, c], n_bins=n_bins, strategy="uniform")
        bins[LABEL_NAMES[c]] = {
            "prob_pred": [float(x) for x in prob_pred],
            "prob_true": [float(x) for x in prob_true],
        }
    return bins


def _max_proba_stats(y_proba: np.ndarray) -> dict[str, float]:
    """En yuksek tahmin edilen sinif olasiliginin dagilimi — 'her zaman %100' sorununu izler."""
    max_p = y_proba.max(axis=1)
    return {
        "mean": float(max_p.mean()),
        "median": float(np.median(max_p)),
        "frac_above_0.99": float((max_p > 0.99).mean()),
        "frac_above_0.999": float((max_p > 0.999).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["both", "before", "after"], default="both",
                         help="Sadece bir asamayi calistir (once olculenleri tekrar etmemek icin)")
    parser.add_argument("--save-models", action="store_true",
                         help="Egitilen modelleri outputs/models/_diag_<tag>.pkl olarak kaydet (kisa-metin tanisi icin)")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    models_cfg = load_yaml("models")
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]

    fused_df = pd.read_parquet(processed_dir / "fused.parquet")
    holdout_ids = set((processed_dir / "holdout_ids.txt").read_text(encoding="utf-8").splitlines())

    dev_df = fused_df[~fused_df["sample_id"].isin(holdout_ids)].reset_index(drop=True)
    hold_df = fused_df[fused_df["sample_id"].isin(holdout_ids)].reset_index(drop=True)

    min_n = dev_df["label"].value_counts().min()
    dev_df = pd.concat(
        [g.sample(n=min_n, random_state=42) for _, g in dev_df.groupby("label")],
        ignore_index=True,
    )
    print(f"[calib] dev (dengeli): {len(dev_df)}, held-out (dogal dagilim): {len(hold_df)}")

    feat_cols = [c for c in dev_df.columns if c not in ("sample_id", "label")]
    X_dev = dev_df[feat_cols].to_numpy(dtype=np.float32)
    y_dev = dev_df["label"].map(LABEL_TO_INT).to_numpy()
    X_hold = hold_df[feat_cols].to_numpy(dtype=np.float32)
    y_hold = hold_df["label"].map(LABEL_TO_INT).to_numpy()

    stage_map = {
        "before": [("before_uncalibrated", False)],
        "after": [("after_calibrated", True)],
        "both": [("before_uncalibrated", False), ("after_calibrated", True)],
    }

    results: dict[str, dict] = {}
    for tag, calibrate in stage_map[args.stage]:
        cfg = copy.deepcopy(models_cfg)
        cfg["common"]["calibrate"] = calibrate
        print(f"\n[calib] === {tag} (calibrate={calibrate}) === gelistirme havuzunda egitiliyor...")
        model = train_final_model(X_dev, y_dev, "stacking", cfg)

        if args.save_models:
            diag_path = PROJECT_ROOT / paths_cfg["models_dir"] / f"_diag_{tag}.pkl"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diag_path, "wb") as f:
                pickle.dump(model, f)
            print(f"[calib] tani modeli kaydedildi -> {diag_path}")

        y_pred = model.predict(X_hold)
        y_proba = model.predict_proba(X_hold)

        acc = float(accuracy_score(y_hold, y_pred))
        macro_f1 = float(f1_score(y_hold, y_pred, average="macro", zero_division=0))
        brier = _multiclass_brier(y_hold, y_proba)
        max_stats = _max_proba_stats(y_proba)
        bins = _reliability_bins(y_hold, y_proba)

        print(f"[calib] {tag}: acc={acc:.4f} macro_f1={macro_f1:.4f} brier_mean={brier['mean']:.4f} "
              f"max_proba_mean={max_stats['mean']:.4f} frac>0.999={max_stats['frac_above_0.999']:.4f}")

        results[tag] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "brier_score": brier,
            "max_proba_stats": max_stats,
            "reliability_bins": bins,
        }

    out_dir = PROJECT_ROOT / paths_cfg["reports_dir"] / "cv_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calibration_before_after_{args.stage}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[calib] -> {out_path}")

    if args.stage == "both":
        md_lines = ["# Kalibrasyon Oncesi/Sonrasi Karsilastirma (Held-out Set)", ""]
        md_lines.append("| Metrik | Once (uncalibrated) | Sonra (isotonic, cv=5) |")
        md_lines.append("|---|---|---|")
        b, a = results["before_uncalibrated"], results["after_calibrated"]
        md_lines.append(f"| Accuracy | {b['accuracy']:.4f} | {a['accuracy']:.4f} |")
        md_lines.append(f"| Macro-F1 | {b['macro_f1']:.4f} | {a['macro_f1']:.4f} |")
        md_lines.append(f"| Brier (ortalama) | {b['brier_score']['mean']:.4f} | {a['brier_score']['mean']:.4f} |")
        for lbl in LABEL_NAMES:
            md_lines.append(f"| Brier ({lbl}) | {b['brier_score'][lbl]:.4f} | {a['brier_score'][lbl]:.4f} |")
        md_lines.append(f"| Ort. max-olasilik | {b['max_proba_stats']['mean']:.4f} | {a['max_proba_stats']['mean']:.4f} |")
        md_lines.append(
            f"| Oran(max-olasilik>0.999) | {b['max_proba_stats']['frac_above_0.999']:.4f} | "
            f"{a['max_proba_stats']['frac_above_0.999']:.4f} |"
        )
        (out_dir / "calibration_before_after.md").write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
