"""Leave-One-Generator-Out (LOGO) Cross-Validation.

Amac: `make_holdout_split.py`'nin StratifiedGroupKFold ile ayirdigi held-out set
sadece "hic gorulmemis DOKUMAN" olcuyor -- ureticiler (Qwen/GPT-4o-mini/
Claude) held-out'ta da egitimde de ayni sekilde temsil. Bu script, n=15/15
canli testte (bkz. proje notlari, GPT-4o-mini 15/15 vs Claude 1/15) elle
gosterilen "hic gorulmemis uretici" senaryosunu buyuk orneklemle ve otomatik
olarak olcer: sirayla her ureticiyi egitimden TAMAMEN cikarir, sadece o
ureticide test eder.

UYARI: Her fold ayri bir stacking modeli egitir (4 base model + kendi ic
5-fold'u) -- 3 uretici icin toplam ~3x egitim suresi. Yerelde
`measure_calibration.py`'nin bir kosusu bile yavas oldugu icin bu script
Colab'da (CPU + Yuksek RAM runtime, GPU degil -- XGBoost/CatBoost/MLP/LogReg
GPU'ya ozel yapilandirilmadi) calistirilmak uzere tasarlandi.

Girdi : data/processed/fused.parquet, configs/models.yaml
Cikti : outputs/reports/cv_results/logo_cv.json
        outputs/reports/cv_results/logo_cv.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.evaluation.generator_eval import (
    add_generator_column,
    format_logo_report,
    run_logo_cv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="fused.parquet yolu")
    parser.add_argument(
        "--tag-suffix", default="", help="Cikti dosya adina eklenecek ek etiket (uzerine yazmamak icin)"
    )
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    models_cfg = load_yaml("models")
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]
    fused_path = Path(args.input) if args.input else processed_dir / "fused.parquet"

    print(f"[logo] girdi: {fused_path}")
    df = pd.read_parquet(fused_path)
    df = add_generator_column(df)
    print(f"[logo] uretici dagilimi:\n{df['generator'].value_counts().to_string()}")

    results = run_logo_cv(df, models_cfg)

    out_dir = PROJECT_ROOT / paths_cfg["reports_dir"] / "cv_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"logo_cv{args.tag_suffix}"
    json_path = out_dir / f"{tag}.json"
    md_path = out_dir / f"{tag}.md"

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(format_logo_report(results), encoding="utf-8")

    print(f"[logo] -> {json_path}")
    print(f"[logo] -> {md_path}")


if __name__ == "__main__":
    main()
