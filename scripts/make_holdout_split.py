"""Kaynak-grubu (doküman/prompt) bazli, gercekten hic dokunulmayacak bir held-out test seti ayirir.

Amac: HPO + model karsilastirmasi (CV) SADECE gelistirme havuzunda yapilir; secilen nihai
model, hicbir asamada (feature-engineering istatistikleri, HPO, CV) gormedigi bu held-out
sette SADECE BIR KEZ degerlendirilir. StratifiedGroupKFold kullanilarak ayni kaynak
dokuman/prompt'tan gelen orneklerin train/held-out arasinda bolunmesi (leakage) onlenir.

Girdi : data/processed/fused.parquet, data/processed/groups.parquet
Cikti : data/processed/holdout_ids.txt  (held-out sample_id listesi, ~%20)
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from humanai_detect.config import PROJECT_ROOT, load_yaml

LABEL_TO_INT = {"human": 0, "ai_raw": 1, "ai_humanized": 2}


def main() -> None:
    paths_cfg = load_yaml("paths")
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]

    fused_df = pd.read_parquet(processed_dir / "fused.parquet")
    groups_df = pd.read_parquet(processed_dir / "groups.parquet")

    df = fused_df[["sample_id", "label"]].merge(groups_df[["sample_id", "group_id"]], on="sample_id", how="left")
    if df["group_id"].isna().any():
        missing = df["group_id"].isna().sum()
        raise ValueError(f"{missing} ornekte group_id eslesmedi (groups.parquet guncel mi?)")

    y = df["label"].map(LABEL_TO_INT).to_numpy()
    groups = df["group_id"].to_numpy()

    # 5 parcaya bol, ilk parcayi (~%20) held-out olarak ayir. StratifiedGroupKFold ayni
    # grubun (dokuman/prompt) train/held-out arasina bolunmesini engeller.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    _, holdout_idx = next(iter(sgkf.split(df, y, groups=groups)))

    holdout_ids = df.iloc[holdout_idx]["sample_id"].tolist()
    holdout_groups = set(df.iloc[holdout_idx]["group_id"])
    dev_groups = set(df.iloc[~df.index.isin(holdout_idx)]["group_id"])
    overlap = holdout_groups & dev_groups

    out_path = processed_dir / "holdout_ids.txt"
    out_path.write_text("\n".join(holdout_ids), encoding="utf-8")

    print(f"[holdout] toplam {len(df)} ornek, {df['group_id'].nunique()} benzersiz grup")
    print(f"[holdout] held-out: {len(holdout_ids)} ornek ({len(holdout_ids)/len(df):.1%}), "
          f"{len(holdout_groups)} grup")
    print(f"[holdout] gelistirme havuzu: {len(df)-len(holdout_ids)} ornek, {len(dev_groups)} grup")
    print(f"[holdout] grup cakismasi (olmali=0): {len(overlap)}")
    for label in LABEL_TO_INT:
        n_dev = ((df["label"] == label) & (~df.index.isin(holdout_idx))).sum()
        n_hold = ((df["label"] == label) & (df.index.isin(holdout_idx))).sum()
        print(f"  {label}: dev={n_dev}, holdout={n_hold}")
    print(f"[holdout] -> {out_path}")


if __name__ == "__main__":
    main()
