"""fusion modullerinin birim testleri: standardize, vif, mutual_info, clustering, early_fusion."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# standardize.py
# ---------------------------------------------------------------------------

class TestZscoreStandardize:
    def test_mean_zero_std_one(self):
        from humanai_detect.fusion.standardize import zscore_standardize
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        Xs, mean, std = zscore_standardize(X)
        assert math.isclose(Xs[:, 0].mean(), 0.0, abs_tol=1e-9)
        assert math.isclose(Xs[:, 0].std(), 1.0, rel_tol=1e-5)

    def test_train_mask_isolates_fit(self):
        from humanai_detect.fusion.standardize import zscore_standardize
        X = np.array([[1.0], [2.0], [100.0]])  # son eleman test
        mask = np.array([True, True, False])
        Xs, mean, std = zscore_standardize(X, train_mask=mask)
        # mean ve std sadece [1,2]'den hesaplandi
        assert math.isclose(mean[0], 1.5)

    def test_constant_column_no_divide_by_zero(self):
        from humanai_detect.fusion.standardize import zscore_standardize
        X = np.array([[5.0], [5.0], [5.0]])
        Xs, _, _ = zscore_standardize(X)
        assert np.all(np.isfinite(Xs))

    def test_returns_tuple_of_three(self):
        from humanai_detect.fusion.standardize import zscore_standardize
        result = zscore_standardize(np.eye(3))
        assert len(result) == 3

    def test_standardize_dispatch_zscore(self):
        from humanai_detect.fusion.standardize import standardize
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Xs, _, _ = standardize(X, method="zscore")
        assert Xs.shape == X.shape

    def test_standardize_dispatch_unknown_raises(self):
        from humanai_detect.fusion.standardize import standardize
        with pytest.raises(ValueError):
            standardize(np.eye(2), method="l2")

    def test_nan_row_does_not_poison_column(self):
        from humanai_detect.fusion.standardize import zscore_standardize
        # 3. satirin 2. sutunu eksik (orn. tek-cumlelik metinde burstiness NaN)
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, np.nan]])
        Xs, mean, std = zscore_standardize(X)
        # mean/std sadece o sutunun eksik-olmayan degerlerinden hesaplanmali
        assert math.isclose(mean[1], 15.0)
        # eksik satirin kendi degeri 0'a (ortalamaya) impute edilmeli, NaN kalmamali
        assert np.all(np.isfinite(Xs))
        assert math.isclose(Xs[2, 1], 0.0, abs_tol=1e-9)
        # diger sutun (NaN icermeyen) bu satirdan etkilenmemeli
        assert math.isclose(Xs[0, 0], (1.0 - 2.0) / std[0], rel_tol=1e-6)

    def test_robust_standardize_nan_aware(self):
        from humanai_detect.fusion.standardize import robust_standardize
        X = np.array([[1.0], [2.0], [3.0], [np.nan]])
        Xs, median, iqr = robust_standardize(X)
        assert math.isclose(median[0], 2.0)
        assert np.all(np.isfinite(Xs))


# ---------------------------------------------------------------------------
# clustering.py
# ---------------------------------------------------------------------------

class TestCosineFeatureClustering:
    def test_identical_features_same_cluster(self):
        from humanai_detect.fusion.clustering import cosine_feature_clustering
        # Iki tamamen ayni sutun -> ayni kumede olmali
        X = np.ones((10, 2))
        labels = cosine_feature_clustering(X, eps=0.1)
        assert labels[0] == labels[1]

    def test_orthogonal_features_different_clusters(self):
        from humanai_detect.fusion.clustering import cosine_feature_clustering
        # Birbirinden tamamen bagimsiz iki ozellik
        X = np.zeros((4, 2))
        X[:2, 0] = 1.0  # ozellik 0: ilk 2 satir 1, diger 0
        X[2:, 1] = 1.0  # ozellik 1: son 2 satir 1, diger 0
        labels = cosine_feature_clustering(X, eps=0.5)
        assert labels[0] != labels[1]

    def test_returns_integer_array(self):
        from humanai_detect.fusion.clustering import cosine_feature_clustering
        X = np.random.default_rng(0).standard_normal((20, 5))
        labels = cosine_feature_clustering(X, eps=0.3)
        assert labels.dtype in (np.int32, np.int64)
        assert len(labels) == 5

    def test_select_representatives_one_per_cluster(self):
        from humanai_detect.fusion.clustering import select_cluster_representatives
        labels = np.array([0, 0, 1, 1, 2])
        X = np.random.default_rng(1).standard_normal((20, 5))
        reps = select_cluster_representatives(X, labels)
        assert len(reps) == 3  # 3 kume -> 3 temsilci
        assert len(set(reps)) == 3  # hepsi farkli indeks


# ---------------------------------------------------------------------------
# early_fusion.py
# ---------------------------------------------------------------------------

class TestFuse:
    def _make_sty(self, n=4):
        return pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "label": ["human"] * n,
            "ttr": np.random.rand(n),
            "burstiness": np.random.rand(n),
        })

    def _make_emb(self, n=4, d=8):
        df = pd.DataFrame(
            np.random.rand(n, d),
            columns=[f"dim_{i}" for i in range(d)],
        )
        df.insert(0, "sample_id", [f"s{i}" for i in range(n)])
        df.insert(1, "label", ["human"] * n)
        return df

    def test_output_shape(self):
        from humanai_detect.fusion.early_fusion import fuse
        sty = self._make_sty(4)
        emb = self._make_emb(4, 8)
        fusion_cfg = {"scaler": "zscore", "embedding_dim_reduction": {"method": "none"}}
        arr, names = fuse(sty, [emb], fusion_cfg)
        # 2 stilometrik + 8 emb boyutu = 10 sutun
        assert arr.shape == (4, 10)
        assert len(names) == 10

    def test_no_embeddings(self):
        from humanai_detect.fusion.early_fusion import fuse
        sty = self._make_sty(3)
        fusion_cfg = {"scaler": "zscore", "embedding_dim_reduction": {"method": "none"}}
        arr, names = fuse(sty, [], fusion_cfg)
        assert arr.shape == (3, 2)

    def test_nan_filled_with_zero(self):
        from humanai_detect.fusion.early_fusion import fuse
        sty = self._make_sty(4)
        sty.loc[0, "ttr"] = float("nan")
        fusion_cfg = {"scaler": "zscore", "embedding_dim_reduction": {"method": "none"}}
        arr, _ = fuse(sty, [], fusion_cfg)
        assert np.all(np.isfinite(arr))

    def test_build_fused_dataframe_columns(self):
        from humanai_detect.fusion.early_fusion import build_fused_dataframe
        sty = self._make_sty(4)
        emb = self._make_emb(4, 4)
        fusion_cfg = {"scaler": "zscore", "embedding_dim_reduction": {"method": "none"}}
        df = build_fused_dataframe(sty, [("bert", emb)], fusion_cfg)
        assert "sample_id" in df.columns
        assert "label" in df.columns
        assert df.shape == (4, 2 + 2 + 4)  # meta + sty + emb


# ---------------------------------------------------------------------------
# length_residualize.py
# ---------------------------------------------------------------------------

class TestLengthResidualizer:
    def _make_df(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        token_counts = rng.integers(20, 900, size=n).astype(float)
        # kl_div_word_freq'i uzunlukla dogrudan (gurultulu) iliskili uret: kisa metinde yuksek
        noise = rng.normal(0, 0.05, size=n)
        kl = 2.0 / np.log1p(token_counts) + noise
        df = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "label": ["human"] * n,
            "kl_div_word_freq": kl,
        })
        return df, token_counts

    def test_residual_removes_length_correlation(self):
        from humanai_detect.fusion.length_residualize import (
            fit_length_residualizer, apply_length_residualizer_df,
        )
        df, token_counts = self._make_df()
        params = fit_length_residualizer(df, token_counts, feature_names=["kl_div_word_freq"])
        residualized = apply_length_residualizer_df(df, token_counts, params)

        corr_before = np.corrcoef(df["kl_div_word_freq"], np.log1p(token_counts))[0, 1]
        corr_after = np.corrcoef(residualized["kl_div_word_freq"], np.log1p(token_counts))[0, 1]
        assert abs(corr_before) > 0.5
        assert abs(corr_after) < 0.1

    def test_train_mask_excludes_holdout_from_fit(self):
        from humanai_detect.fusion.length_residualize import fit_length_residualizer
        df, token_counts = self._make_df()
        mask = np.array([True] * 150 + [False] * 50)
        params_masked = fit_length_residualizer(df, token_counts, feature_names=["kl_div_word_freq"], train_mask=mask)
        params_full = fit_length_residualizer(df, token_counts, feature_names=["kl_div_word_freq"])
        # farkli alt kumelerle fit edildigi icin katsayilar birebir ayni olmamali
        assert params_masked["kl_div_word_freq"] != params_full["kl_div_word_freq"]

    def test_missing_feature_ignored(self):
        from humanai_detect.fusion.length_residualize import fit_length_residualizer
        df, token_counts = self._make_df()
        params = fit_length_residualizer(df, token_counts, feature_names=["nonexistent_feature"])
        assert params == {}

    def test_apply_dict_single_sample(self):
        from humanai_detect.fusion.length_residualize import apply_length_residualizer_dict
        params = {"burstiness": {"slope": 0.5, "intercept": -1.0}}
        feats = {"burstiness": 0.3, "ttr": 0.7}
        out = apply_length_residualizer_dict(feats, token_count=99, params=params)
        log_tok = np.log1p(99)
        expected = 0.3 - (0.5 * log_tok - 1.0)
        assert math.isclose(out["burstiness"], expected, rel_tol=1e-9)
        assert out["ttr"] == 0.7  # ilgisiz feature degismemeli

    def test_apply_dict_skips_nan(self):
        from humanai_detect.fusion.length_residualize import apply_length_residualizer_dict
        params = {"kl_div_word_freq": {"slope": 1.0, "intercept": 0.0}}
        feats = {"kl_div_word_freq": float("nan")}
        out = apply_length_residualizer_dict(feats, token_count=50, params=params)
        assert math.isnan(out["kl_div_word_freq"])
