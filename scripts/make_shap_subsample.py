"""SHAP analizi icin sinif-dengeli, kucuk bir alt-ornek olusturur.

Tam veri (11896 ornek x 815 ozellik x 3 sinif) ile TreeExplainer 25+ dakikada
bitmiyor (bkz. proje notlari). Onceki denemede alt-orneklem ad-hoc bir
komutla, `label`/`generator` kolonlari olmadan, hicbir script'e kaydedilmeden
uretilmisti -- bu yuzden hangi ornegin hangi sinifa/ureticiye ait oldugu
kayboldu ve uretici-bazli SHAP kiyasi (explain_by_generator.py) yapilamiyordu.
Bu script ayni islemi tekrarlanabilir/izlenebilir hale getirir.

Girdi : data/processed/fused.parquet
Cikti : data/processed/shap_subsample.parquet  (sample_id + label + generator + ozellikler)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.evaluation.generator_eval import add_generator_column


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="fused.parquet yolu")
    parser.add_argument(
        "--n-per-class", type=int, default=700, help="Sinif basina alt-ornek sayisi (varsayilan 700 ~ 2100 toplam)"
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default=None, help="Cikti parquet yolu")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]
    fused_path = Path(args.input) if args.input else processed_dir / "fused.parquet"

    print(f"[shap-subsample] girdi: {fused_path}")
    df = pd.read_parquet(fused_path)
    df = add_generator_column(df)

    parts = []
    for label, g in df.groupby("label"):
        n = min(len(g), args.n_per_class)
        parts.append(g.sample(n=n, random_state=args.random_state))
    subsample = pd.concat(parts, ignore_index=True)

    out_path = Path(args.output) if args.output else processed_dir / "shap_subsample.parquet"
    subsample.to_parquet(out_path, index=False)

    print(f"[shap-subsample] {len(df)} -> {len(subsample)} ornek")
    print(subsample.groupby(["label", "generator"], observed=True).size().to_string())
    print(f"[shap-subsample] -> {out_path}")


if __name__ == "__main__":
    main()
