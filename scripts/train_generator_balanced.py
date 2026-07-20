"""Uretici-dengeli egitim deneyi: ai_raw/ai_humanized havuzundaki asiri Qwen
temsilini bir hedef sayiya indirip, ayni LOGO degerlendirmesini tekrarlar --
"dengesiz uretici payi zayif genellemenin asil sebebi mi" hipotezini dogrudan,
ucuza (veri kaybi yok, sadece alt-orneklem; kucuk ureticiler ASLA yukari-
orneklenmez/tekrarlanmaz) test eder.

Onkosul: once `scripts/logo_cv.py` calistirilip `logo_cv.json` uretilmis olmali
(karsilastirma bunun uzerine kurulu).

Girdi : data/processed/fused.parquet, configs/models.yaml,
        outputs/reports/cv_results/logo_cv.json (baseline)
Cikti : outputs/reports/cv_results/logo_cv_generator_balanced.json / .md
        outputs/reports/cv_results/generator_balance_comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.evaluation.generator_eval import (
    NON_HUMAN_GENERATORS,
    add_generator_column,
    format_logo_report,
    run_logo_cv,
)


def downsample_generators(
    df: pd.DataFrame, target_per_generator: int, random_state: int = 42
) -> pd.DataFrame:
    """(label, generator) ikilisi basina ornek sayisini target_per_generator'a indirir.

    SADECE target_per_generator'i ASAN gruplar alt-orneklenir; altinda kalan
    kucuk ureticiler (orn. Claude ~209) OLDUGU GIBI birakilir -- asla yukari-
    orneklenip tekrarlanmaz (tekrar, ayni ornegin coklanmasindan ibaret sahte
    cesitlilik yaratirdi). human sinifi hic dokunulmaz (ureticisi yok).
    """
    parts = []
    for (label, gen), g in df.groupby(["label", "generator"], observed=True):
        if label == "human":
            parts.append(g)
            continue
        n = min(len(g), target_per_generator)
        parts.append(g.sample(n=n, random_state=random_state))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="fused.parquet yolu")
    parser.add_argument(
        "--target-per-generator",
        type=int,
        default=1000,
        help=(
            "ai_raw/ai_humanized icin uretici basina hedef ornek sayisi (varsayilan 1000, "
            "GPT-4o-mini'nin dogal payina yakin) -- Qwen'in ~6217'lik toplami bu sayiya "
            "indirilir, Claude'un ~418'lik toplami zaten altinda oldugu icin dokunulmaz."
        ),
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Karsilastirma icin logo_cv.json yolu (varsayilan: outputs/reports/cv_results/logo_cv.json)",
    )
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    models_cfg = load_yaml("models")
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]
    fused_path = Path(args.input) if args.input else processed_dir / "fused.parquet"

    print(f"[gen-balanced] girdi: {fused_path}")
    df = pd.read_parquet(fused_path)
    df = add_generator_column(df)
    print(f"[gen-balanced] once (label, generator):\n{df.groupby(['label', 'generator'], observed=True).size().to_string()}")

    df_bal = downsample_generators(df, args.target_per_generator)
    print(f"[gen-balanced] sonra (label, generator):\n{df_bal.groupby(['label', 'generator'], observed=True).size().to_string()}")

    results = run_logo_cv(df_bal, models_cfg)

    out_dir = PROJECT_ROOT / paths_cfg["reports_dir"] / "cv_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "logo_cv_generator_balanced.json"
    md_path = out_dir / "logo_cv_generator_balanced.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(format_logo_report(results), encoding="utf-8")
    print(f"[gen-balanced] -> {json_path}")
    print(f"[gen-balanced] -> {md_path}")

    baseline_path = Path(args.baseline) if args.baseline else out_dir / "logo_cv.json"
    if not baseline_path.exists():
        print(f"[gen-balanced] baseline bulunamadi ({baseline_path}) -- once scripts/logo_cv.py "
              f"calistirin, karsilastirma tablosu atlandi.")
        return

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    lines = [
        "# Uretici-Dengeleme Karsilastirmasi (Baseline vs Dengeli)\n",
        f"Baseline: tum ureticiler dogal payiyla (Qwen ~%89 baskin). Dengeli: ai_raw/"
        f"ai_humanized uretici basina en fazla {args.target_per_generator} ornekle "
        f"sinirlandi (Qwen alt-orneklendi, kucuk ureticiler dokunulmadi).\n",
        "| Uretici (egitim-disi) | Katı dogruluk (baseline) | Katı dogruluk (dengeli) | Fark |",
        "|---|---|---|---|",
    ]
    for gen in NON_HUMAN_GENERATORS:
        if gen not in results or gen not in baseline:
            continue
        b_acc = baseline[gen]["generator_only_accuracy_strict"]
        a_acc = results[gen]["generator_only_accuracy_strict"]
        lines.append(f"| {gen} | {b_acc:.4f} | {a_acc:.4f} | {a_acc - b_acc:+.4f} |")

    comparison_path = out_dir / "generator_balance_comparison.md"
    comparison_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[gen-balanced] karsilastirma -> {comparison_path}")


if __name__ == "__main__":
    main()
