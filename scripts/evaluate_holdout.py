"""Secilen nihai modeli gelistirme havuzunda (holdout haric) egitir, held-out sette
SADECE BIR KEZ degerlendirir. Bu, CV/HPO'nun hicbir asamasinda gormedigi gercekten
bagimsiz bir test seti oldugu icin raporlanacak 'resmi' sayi budur.

Girdi : data/processed/fused.parquet, data/processed/holdout_ids.txt, configs/models.yaml
Cikti : outputs/reports/cv_results/<model>_holdout_eval.json/.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score,
)

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.models.train import train_final_model
from humanai_detect.models.hierarchical import train_final_hierarchical

LABEL_NAMES = ["human", "ai_raw", "ai_humanized"]
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(LABEL_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["xgboost", "catboost", "mlp", "logreg", "stacking", "hierarchical"],
                         default="stacking")
    parser.add_argument("--balance-dev", action="store_true", default=True,
                         help="Gelistirme havuzunu dengele (varsayilan: acik)")
    parser.add_argument("--balance-holdout", action="store_true",
                         help="Held-out seti de dengele (varsayilan: dogal dagilimla degerlendir)")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    models_cfg = load_yaml("models")
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]

    fused_df = pd.read_parquet(processed_dir / "fused.parquet")
    holdout_path = processed_dir / "holdout_ids.txt"
    if not holdout_path.exists():
        print("[holdout-eval] holdout_ids.txt yok. Once make_holdout_split.py calistirin.")
        return
    holdout_ids = set(holdout_path.read_text(encoding="utf-8").splitlines())

    dev_df = fused_df[~fused_df["sample_id"].isin(holdout_ids)].reset_index(drop=True)
    hold_df = fused_df[fused_df["sample_id"].isin(holdout_ids)].reset_index(drop=True)
    print(f"[holdout-eval] gelistirme havuzu: {len(dev_df)}, held-out: {len(hold_df)}")

    if args.balance_dev:
        min_n = dev_df["label"].value_counts().min()
        dev_df = pd.concat(
            [g.sample(n=min_n, random_state=42) for _, g in dev_df.groupby("label")],
            ignore_index=True,
        )
        print(f"[holdout-eval] dev dengelendi: her siniftan {min_n} ornek (toplam {len(dev_df)})")

    if args.balance_holdout:
        min_n = hold_df["label"].value_counts().min()
        hold_df = pd.concat(
            [g.sample(n=min_n, random_state=42) for _, g in hold_df.groupby("label")],
            ignore_index=True,
        )
        print(f"[holdout-eval] held-out dengelendi: her siniftan {min_n} ornek (toplam {len(hold_df)})")
    else:
        print(f"[holdout-eval] held-out dogal dagilimla degerlendiriliyor: "
              f"{dict(hold_df['label'].value_counts())}")

    feat_cols = [c for c in dev_df.columns if c not in ("sample_id", "label")]
    X_dev = dev_df[feat_cols].to_numpy(dtype=np.float32)
    y_dev = dev_df["label"].map(LABEL_TO_INT).to_numpy()
    X_hold = hold_df[feat_cols].to_numpy(dtype=np.float32)
    y_hold = hold_df["label"].map(LABEL_TO_INT).to_numpy()

    common = models_cfg.get("common", {})
    print(f"[holdout-eval] {args.model} gelistirme havuzunda egitiliyor ({len(X_dev)} ornek)...")
    if args.model == "stacking":
        model = train_final_model(X_dev, y_dev, "stacking", models_cfg)
    elif args.model == "hierarchical":
        hier_cfg = models_cfg.get("hierarchical", {})
        stage_a_name = hier_cfg.get("stage_a_model", "xgboost")
        stage_b_name = hier_cfg.get("stage_b_model", "xgboost")
        stage_a_params = {**common, **models_cfg.get(stage_a_name, {})}
        stage_b_params = {**common, **models_cfg.get(stage_b_name, {})}
        model = train_final_hierarchical(X_dev, y_dev, stage_a_name, stage_a_params, stage_b_name, stage_b_params)
    else:
        model_params = {**common, **models_cfg.get(args.model, {})}
        model = train_final_model(X_dev, y_dev, args.model, model_params)

    print("[holdout-eval] held-out sette degerlendiriliyor (TEK SEFERLIK)...")
    y_pred = model.predict(X_hold)

    per_class_f1 = f1_score(y_hold, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_hold, y_pred, labels=[0, 1, 2])

    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_hold)
            roc_auc = float(roc_auc_score(y_hold, y_proba, multi_class="ovr"))
        except Exception as e:
            print(f"[holdout-eval] ROC-AUC hesaplanamadi: {e}")

    result = {
        "model": args.model,
        "n_dev": int(len(X_dev)),
        "n_holdout": int(len(X_hold)),
        "accuracy": float(accuracy_score(y_hold, y_pred)),
        "macro_f1": float(f1_score(y_hold, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_hold, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {LABEL_NAMES[i]: float(per_class_f1[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABEL_NAMES,
        "roc_auc_ovr": roc_auc,
    }

    print(f"[holdout-eval] Accuracy={result['accuracy']:.4f}  Macro-F1={result['macro_f1']:.4f}")
    print(f"[holdout-eval] Per-class F1: {result['per_class_f1']}")
    print(f"[holdout-eval] Confusion matrix ({LABEL_NAMES}):\n{cm}")

    report_dir = PROJECT_ROOT / paths_cfg["reports_dir"] / "cv_results"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"{args.model}_holdout_eval.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[holdout-eval] -> {out_path}")

    md_lines = [
        f"# Held-out Test Sonuclari — {args.model}",
        f"\nGelistirme havuzu: {result['n_dev']} ornek, Held-out (hic gorulmemis): {result['n_holdout']} ornek",
        f"\n**Accuracy:** {result['accuracy']:.4f}",
        f"**Macro-F1:** {result['macro_f1']:.4f}",
        f"**Weighted-F1:** {result['weighted_f1']:.4f}",
        f"**ROC-AUC (OvR):** {roc_auc:.4f}" if roc_auc is not None else "",
        "\n| Sinif | F1 |",
        "|---|---|",
    ]
    for lbl in LABEL_NAMES:
        md_lines.append(f"| {lbl} | {result['per_class_f1'][lbl]:.4f} |")
    md_lines.append("\n## Confusion Matrix (satir=gercek, sutun=tahmin)\n")
    md_lines.append("| | " + " | ".join(LABEL_NAMES) + " |")
    md_lines.append("|---|" + "---|" * 3)
    for i, lbl in enumerate(LABEL_NAMES):
        md_lines.append(f"| **{lbl}** | " + " | ".join(str(x) for x in cm[i]) + " |")
    (report_dir / f"{args.model}_holdout_eval.md").write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
