"""Stilometrik ozellikler + transformer embedding'lerini birlestiren early-fusion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .standardize import standardize


def fuse(
    stylometric_df: pd.DataFrame,
    embedding_dfs: list[pd.DataFrame],
    fusion_cfg: dict,
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Stilometrik ozellikler ile embedding matrislerini standartlastirip birlestir.

    stylometric_df : ['sample_id', 'label', feat1, feat2, ...] — Asama 3 ciktisi
    embedding_dfs  : her biri ['sample_id', 'label', dim_0, dim_1, ...] — Asama 4 ciktisi
    fusion_cfg     : configs/fusion.yaml iceriginin Python sozlugu
    train_mask     : egitim orneklerinin boolean maskesi (fit icin); None -> tum veri

    Donus:
        fused_matrix  : float32 ndarray [n_samples, n_total_features]
        feature_names : her sutunun isim listesi
    """
    meta_cols = {"sample_id", "label"}

    # --- Stilometrik ozellikler ---
    sty_cols = [c for c in stylometric_df.columns if c not in meta_cols]
    sty_arr = stylometric_df[sty_cols].to_numpy(dtype=np.float32)
    sty_std, _, _ = standardize(sty_arr, method=fusion_cfg.get("scaler", "zscore"), train_mask=train_mask)
    sty_std = np.nan_to_num(sty_std, nan=0.0)  # NaN referans ozellikleri -> 0

    parts = [sty_std]
    names = list(sty_cols)

    # --- Embedding boyutu azaltma (PCA, opsiyonel) ---
    dim_red_cfg = fusion_cfg.get("embedding_dim_reduction", {})
    dim_red_method = dim_red_cfg.get("method", "none")
    n_components = dim_red_cfg.get("n_components", 128)

    for emb_df in embedding_dfs:
        # Hangi model? (sample_id ve label haricindeki sutunlar dim_*)
        emb_cols = [c for c in emb_df.columns if c not in meta_cols]
        model_name = _infer_model_name(emb_cols)
        emb_arr = emb_df[emb_cols].to_numpy(dtype=np.float32)
        emb_std, _, _ = standardize(emb_arr, method="zscore", train_mask=train_mask)

        if dim_red_method == "pca" and n_components < emb_arr.shape[1]:
            emb_std = _apply_pca(emb_std, n_components, train_mask)
            emb_names = [f"{model_name}_pc{i}" for i in range(n_components)]
        else:
            emb_names = [f"{model_name}_{c}" for c in emb_cols]

        parts.append(emb_std)
        names.extend(emb_names)

    fused = np.hstack(parts).astype(np.float32)
    return fused, names


def _infer_model_name(dim_cols: list[str]) -> str:
    """Sutun adlarindan model ismini cikar (berturk / roberta / genel)."""
    # Embedding sutunlari 'dim_0', 'dim_1'... seklindeise dosya adindan gelir
    # Bu fonksiyon cagiran tarafindan override edilebilir; burda fallback 'emb' kullan.
    return "emb"


def _apply_pca(arr: np.ndarray, n_components: int, train_mask: np.ndarray | None) -> np.ndarray:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=42)
    fit_data = arr[train_mask] if train_mask is not None else arr
    pca.fit(fit_data)
    return pca.transform(arr)


def build_fused_dataframe(
    stylometric_df: pd.DataFrame,
    embedding_dfs: list[tuple[str, pd.DataFrame]],
    fusion_cfg: dict,
    train_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """fuse() sarmalayicisi: metadata ile birlikte tam DataFrame dondurur.

    embedding_dfs: [(model_name, df), ...] listesi — model ismi sutun oneki olarak kullanilir.
    """
    named_emb_dfs = []
    for model_name, emb_df in embedding_dfs:
        # Sutun adlarini model ismini icerecek sekilde yeniden adlandir
        rename = {c: f"{model_name}_{c}" for c in emb_df.columns if c not in {"sample_id", "label"}}
        named_emb_dfs.append(emb_df.rename(columns=rename))

    fused_arr, feat_names = fuse(
        stylometric_df,
        named_emb_dfs,
        fusion_cfg,
        train_mask=train_mask,
    )

    df = pd.DataFrame(fused_arr, columns=feat_names)
    df.insert(0, "label", stylometric_df["label"].values)
    df.insert(0, "sample_id", stylometric_df["sample_id"].values)
    return df
