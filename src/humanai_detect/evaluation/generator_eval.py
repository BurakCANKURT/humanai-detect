"""Leave-One-Generator-Out (LOGO) degerlendirme yardimcilari.

Amac: `make_holdout_split.py`'nin StratifiedGroupKFold'u sadece dokuman/prompt
sizintisini engelliyor -- held-out'taki tum ureticiler egitimde de gorulmus
oluyor (in-distribution genelleme). Bu modul, n=15/15 canli testte elle
gosterilen "hic gorulmemis uretici" senaryosunu (bkz. proje notlari:
GPT-4o-mini 15/15 vs Claude 1/15) buyuk orneklemle ve otomatik olarak
tekrarlar: sirayla her ureticiyi (qwen/gpt4o_mini/claude_sonnet5) EGITIMDEN
TAMAMEN cikarip sadece o ureticide test eder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from humanai_detect.evaluation.metrics import compute_metrics
from humanai_detect.models.factory import build_model
from humanai_detect.utils.generator_id import infer_generator

LABEL_NAMES = ["human", "ai_raw", "ai_humanized"]
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(LABEL_NAMES)}

NON_HUMAN_GENERATORS = ["qwen", "gpt4o_mini", "claude_sonnet5"]


def add_generator_column(df: pd.DataFrame) -> pd.DataFrame:
    """sample_id'den turetilen 'generator' kolonunu ekler (yoksa)."""
    if "generator" in df.columns:
        return df
    df = df.copy()
    df["generator"] = df["sample_id"].map(infer_generator)
    return df


def split_human(
    df: pd.DataFrame, test_frac: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Human orneklerini sabit bir train/test ayrimina boler.

    Insanin 'ureticisi' olmadigi icin LOGO fold'lari arasinda degismez -- her
    fold'da ayni human-test kullanilir, boylece fold'lar arasi fark SADECE
    egitim-disi birakilan ureticiden kaynaklanir.
    """
    human = df[df["label"] == "human"]
    test_human = human.sample(frac=test_frac, random_state=random_state)
    train_human = human.drop(test_human.index)
    return train_human, test_human


def build_logo_fold(
    df: pd.DataFrame,
    held_out_generator: str,
    train_human: pd.DataFrame,
    test_human: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """held_out_generator DISINDAKI uretici verisiyle egitim, held_out_generator +
    test_human ile test seti olusturur."""
    non_human = df[df["label"] != "human"]
    train_non_human = non_human[non_human["generator"] != held_out_generator]
    test_non_human = non_human[non_human["generator"] == held_out_generator]

    train_df = pd.concat([train_human, train_non_human], ignore_index=True)
    test_df = pd.concat([test_human, test_non_human], ignore_index=True)
    return train_df, test_df


def balance_by_label(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Her sinifi en kucuk sinifin sayisina indirger (asiri-temsili sinif egitimi
    domine etmesin diye) -- measure_calibration.py'deki ayni mantik."""
    min_n = df["label"].value_counts().min()
    return pd.concat(
        [g.sample(n=min_n, random_state=random_state) for _, g in df.groupby("label")],
        ignore_index=True,
    )


def _xy(df: pd.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].map(LABEL_TO_INT).to_numpy()
    return X, y


def run_logo_cv(
    df: pd.DataFrame, models_cfg: dict[str, Any], random_state: int = 42
) -> dict[str, Any]:
    """Her uretici icin sirayla egitim-disi birak, kalanla (dengeli) egit, o
    ureticide + sabit human-test'te test et.

    Donen sozlukteki her uretici anahtari icin:
        overall_metrics                : test seti butununde (human_test + o uretici) 3-sinif metrikler
        generator_only_accuracy_strict : SADECE o ureticinin orneklerinde katı (3-sinif) dogruluk
        generator_only_accuracy_binary : SADECE o ureticinin orneklerinde ikili (human-mi-degil-mi) dogruluk
    """
    df = add_generator_column(df)
    feat_cols = [c for c in df.columns if c not in ("sample_id", "label", "generator")]

    train_human, test_human = split_human(df, random_state=random_state)

    results: dict[str, Any] = {}
    for gen in NON_HUMAN_GENERATORS:
        if (df["generator"] == gen).sum() == 0:
            print(f"[logo] {gen}: veri yok, atlaniyor")
            continue

        train_df, test_df = build_logo_fold(df, gen, train_human, test_human)
        train_df = balance_by_label(train_df, random_state=random_state)

        X_train, y_train = _xy(train_df, feat_cols)
        X_test, y_test = _xy(test_df, feat_cols)

        model = build_model("stacking", models_cfg)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_proba=y_proba, label_names=LABEL_NAMES)

        gen_mask = (test_df["generator"] == gen).to_numpy()
        gen_acc_strict = float((y_pred[gen_mask] == y_test[gen_mask]).mean())
        y_test_bin = (y_test[gen_mask] != LABEL_TO_INT["human"]).astype(int)
        y_pred_bin = (y_pred[gen_mask] != LABEL_TO_INT["human"]).astype(int)
        gen_acc_binary = float((y_test_bin == y_pred_bin).mean())

        results[gen] = {
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_test_generator_only": int(gen_mask.sum()),
            "overall_metrics": metrics,
            "generator_only_accuracy_strict": gen_acc_strict,
            "generator_only_accuracy_binary": gen_acc_binary,
        }
        print(
            f"[logo] egitim-disi={gen}: n_train={len(train_df)} n_test_gen={int(gen_mask.sum())} "
            f"katı_acc={gen_acc_strict:.4f} ikili_acc={gen_acc_binary:.4f} "
            f"genel_macro_f1={metrics['macro_f1']:.4f}"
        )

    return results


def format_logo_report(results: dict[str, Any]) -> str:
    lines = [
        "# Leave-One-Generator-Out (LOGO) Degerlendirme Raporu\n",
        "Her satirda belirtilen uretici EGITIMDEN TAMAMEN CIKARILDI (o ureticinin "
        "ne ai_raw ne ai_humanized ornegi egitimde goruldu), sadece testte "
        "kullanildi -- gercek bir 'hic gorulmemis uretici' (zero-shot cross-generator) "
        "senaryosu. Bu, n=15/15 canli testin (bkz. proje notlari) buyuk-orneklemli, "
        "otomatik/tekrarlanabilir hali.\n",
        "| Egitim-disi birakilan uretici | n (test, sadece bu uretici) | "
        "Katı dogruluk (3-sinif) | Ikili dogruluk (AI mi degil mi) | "
        "Genel Macro-F1 (human-test + bu uretici) |",
        "|---|---|---|---|---|",
    ]
    for gen, r in results.items():
        lines.append(
            f"| {gen} | {r['n_test_generator_only']} | "
            f"{r['generator_only_accuracy_strict']:.4f} | "
            f"{r['generator_only_accuracy_binary']:.4f} | "
            f"{r['overall_metrics']['macro_f1']:.4f} |"
        )
    return "\n".join(lines)
