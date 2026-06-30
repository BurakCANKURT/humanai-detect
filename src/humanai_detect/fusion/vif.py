"""Variance Inflation Factor (VIF) ile coklu dogrusal bagimlilik analizi ve filtreleme."""

from __future__ import annotations

import pandas as pd


def compute_vif(features_df: pd.DataFrame) -> pd.DataFrame:
    """Her ozellik icin VIF degerini hesaplar.

    VIF > 10 -> ciddi coklu dogrusal bagimlilik (genellikle o ozellik cikarilir).
    NaN iceren sutunlar hesaplamadan once dusurulur.

    Donus: ['feature', 'vif'] sutunlu DataFrame, azalan VIF sirali.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    df = features_df.dropna(axis=1).select_dtypes(include="number")
    if df.shape[1] < 2:
        return pd.DataFrame({"feature": df.columns.tolist(), "vif": [float("nan")] * df.shape[1]})

    arr = df.values
    records = []
    for i, col in enumerate(df.columns):
        try:
            vif = variance_inflation_factor(arr, i)
        except Exception:
            vif = float("nan")
        records.append({"feature": col, "vif": vif})

    return pd.DataFrame(records).sort_values("vif", ascending=False).reset_index(drop=True)


def filter_by_vif(features_df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """VIF esiginin uzerindeki ozellikleri iteratif olarak cikarir.

    Her adimda en yuksek VIF'li sutun cikarilir, VIF yeniden hesaplanir.
    Donus: kalan sutunlar, daraltilamayacak kadar az sutun kalirsa oldugu gibi doner.
    """
    df = features_df.copy()
    while True:
        if df.shape[1] < 2:
            break
        vif_df = compute_vif(df)
        max_row = vif_df.iloc[0]
        if max_row["vif"] > threshold and not pd.isna(max_row["vif"]):
            df = df.drop(columns=[max_row["feature"]])
        else:
            break
    return df
