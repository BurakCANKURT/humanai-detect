"""Uretici-bazli SHAP kiyaslamasi.

Soru: genel top-N SHAP ozelligi (explain.py'nin ciktisi) HER ureticide
(Qwen/GPT-4o-mini/Claude) tutarli sekilde onemli mi (genellenebilir AI-sinyali),
yoksa sadece bir ureticiye mi ozgu (o ureticinin imzasi -- bkz. proje
notlarindaki kl_div_word_freq ornegi: Qwen'de guclu AI-sinyali (+0.467),
GPT-4o-mini'de neredeyse sifir (-0.022))?

Onkosul: once `scripts/make_shap_subsample.py` ile label+generator kolonlarini
koruyan bir alt-ornek uretilmis olmali.

Girdi : outputs/models/<model>.pkl, data/processed/shap_subsample.parquet
Cikti : outputs/reports/shap_by_generator.md
        outputs/reports/shap_by_generator.csv
        outputs/figures/shap/generator_comparison_top15.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.evaluation.generator_eval import NON_HUMAN_GENERATORS, add_generator_column
from humanai_detect.explainability.shap_analysis import _get_explainer


def _shap_values_2d(shap_values) -> np.ndarray:
    """Explanation/ndarray'i (n_samples, n_features) matrisine indirger (cok sinifli
    ise sinif ekseninde ortalama alinir)."""
    vals = np.asarray(shap_values.values) if hasattr(shap_values, "values") else np.asarray(shap_values)
    if vals.ndim == 3:  # (n_samples, n_features, n_classes)
        vals = np.abs(vals).mean(axis=2)
    else:
        vals = np.abs(vals)
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="SHAP icin .pkl model dosyasi")
    parser.add_argument("--input", default=None, help="shap_subsample.parquet yolu")
    parser.add_argument("--top-n", type=int, default=20, help="Karsilastirilacak genel top-N ozellik sayisi")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]
    sub_path = Path(args.input) if args.input else processed_dir / "shap_subsample.parquet"

    if not sub_path.exists():
        print(f"[explain-gen] {sub_path} bulunamadi. Once scripts/make_shap_subsample.py calistirin.")
        return

    df = pd.read_parquet(sub_path)
    df = add_generator_column(df)

    model_path = Path(args.model)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    feat_cols = [c for c in df.columns if c not in ("sample_id", "label", "generator")]
    X = df[feat_cols].to_numpy(dtype=np.float32)

    print(f"[explain-gen] TreeExplainer, {len(X)} ornek x {len(feat_cols)} ozellik...")
    explainer = _get_explainer(model, X)
    shap_values = explainer(X)
    vals = _shap_values_2d(shap_values)  # (n_samples, n_features)

    overall_mean = vals.mean(axis=0)
    overall_rank = pd.Series(overall_mean, index=feat_cols).sort_values(ascending=False)
    top_features = overall_rank.head(args.top_n).index.tolist()

    per_gen_mean: dict[str, pd.Series] = {}
    for gen in NON_HUMAN_GENERATORS:
        mask = (df["generator"] == gen).to_numpy()
        if mask.sum() == 0:
            print(f"[explain-gen] {gen}: alt-orneklemde ornek yok, atlaniyor")
            continue
        per_gen_mean[gen] = pd.Series(vals[mask].mean(axis=0), index=feat_cols)

    rows = []
    for feat in top_features:
        row: dict[str, object] = {"feature": feat, "overall_mean_abs_shap": float(overall_rank[feat])}
        ranks_present = []
        for gen, series in per_gen_mean.items():
            row[f"mean_abs_shap_{gen}"] = float(series[feat])
            gen_rank = int(series.rank(ascending=False)[feat])
            row[f"rank_{gen}"] = gen_rank
            ranks_present.append(gen_rank <= args.top_n)
        row["consistent_across_generators"] = bool(ranks_present) and all(ranks_present)
        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    report_dir = PROJECT_ROOT / paths_cfg["reports_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "shap_by_generator.csv"
    comparison_df.to_csv(csv_path, index=False, encoding="utf-8")

    n_consistent = int(comparison_df["consistent_across_generators"].sum())
    gen_names = list(per_gen_mean.keys())
    md_lines = [
        "# Uretici-Bazli SHAP Kiyaslamasi\n",
        f"Genel top-{args.top_n} ozellikten **{n_consistent}** tanesi HER ureticide "
        f"(Qwen, GPT-4o-mini, Claude/Sonnet-5) de kendi top-{args.top_n}'inde -- bunlar "
        "genellenebilir/uretici-bagimsiz AI-sinyali adaylaridir. Digerleri en az bir "
        "ureticide top-N disinda -- o ureticiye ozgu bir imza olabilir (bkz. proje "
        "notlarindaki kl_div_word_freq: Qwen'de +0.467 guclu AI-sinyali, GPT-4o-mini'de "
        "-0.022 neredeyse sifir).\n",
        "| Ozellik | Genel Ort.|SHAP| | " + " | ".join(f"{g} rank" for g in gen_names) + " | Tum ureticilerde top-N mi? |",
        "|---|---|" + "---|" * len(gen_names) + "---|",
    ]
    for _, r in comparison_df.iterrows():
        rank_cells = " | ".join(str(int(r[f"rank_{g}"])) for g in gen_names)
        md_lines.append(
            f"| {r['feature']} | {r['overall_mean_abs_shap']:.5f} | {rank_cells} | "
            f"{'EVET' if r['consistent_across_generators'] else 'hayir'} |"
        )
    md_path = report_dir / "shap_by_generator.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[explain-gen] -> {md_path}")
    print(f"[explain-gen] -> {csv_path}")

    _plot_comparison(comparison_df, gen_names, paths_cfg)


def _plot_comparison(comparison_df: pd.DataFrame, generators: list[str], paths_cfg: dict) -> None:
    import matplotlib.pyplot as plt

    top15 = comparison_df.head(15).iloc[::-1]  # yatay barda en onemli en ustte gorunsun
    n_gen = max(len(generators), 1)
    bar_h = 0.8 / n_gen

    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = np.arange(len(top15))
    for i, gen in enumerate(generators):
        vals = top15[f"mean_abs_shap_{gen}"].to_numpy()
        ax.barh(y_pos + i * bar_h, vals, height=bar_h, label=gen)

    ax.set_yticks(y_pos + bar_h * (n_gen - 1) / 2)
    ax.set_yticklabels(top15["feature"])
    ax.set_xlabel("Ortalama |SHAP|")
    ax.set_title("Uretici Bazinda Ozellik Onemi (top-15)")
    ax.legend()
    plt.tight_layout()

    fig_dir = PROJECT_ROOT / paths_cfg["figures_dir"] / "shap"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "generator_comparison_top15.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain-gen] -> {out_path}")


if __name__ == "__main__":
    main()
