"""Asama 4 embedding modullerinin birim testleri.

Gercek model indirmesi gerektiren testler atlanir (torch/transformers mock ile).
Anisotropy ve pooling testleri saf-numpy ile calisir.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# anisotropy.py — saf numpy, model gerektirmez
# ---------------------------------------------------------------------------

class TestComputeAnisotropy:
    def test_identical_vectors_high_anisotropy(self):
        from humanai_detect.embeddings.anisotropy import compute_anisotropy
        vecs = np.ones((5, 4), dtype=float)
        result = compute_anisotropy(vecs)
        assert math.isclose(result, 1.0, rel_tol=1e-5)

    def test_orthogonal_vectors_near_zero(self):
        from humanai_detect.embeddings.anisotropy import compute_anisotropy
        # 4 ortogonal birim vektor
        vecs = np.eye(4)
        result = compute_anisotropy(vecs)
        assert math.isclose(result, 0.0, abs_tol=1e-9)

    def test_single_vector_returns_zero(self):
        from humanai_detect.embeddings.anisotropy import compute_anisotropy
        assert compute_anisotropy(np.array([[1.0, 2.0, 3.0]])) == 0.0

    def test_range(self):
        from humanai_detect.embeddings.anisotropy import compute_anisotropy
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((20, 64))
        result = compute_anisotropy(vecs)
        assert -1.0 <= result <= 1.0


class TestPrincipalDirectionCollapse:
    def test_all_same_full_collapse(self):
        from humanai_detect.embeddings.anisotropy import principal_direction_collapse
        vecs = np.tile([1.0, 0.0, 0.0], (10, 1))
        result = principal_direction_collapse(vecs)
        assert math.isclose(result, 1.0, rel_tol=1e-5)

    def test_identity_matrix_uniform(self):
        from humanai_detect.embeddings.anisotropy import principal_direction_collapse
        vecs = np.eye(4)
        result = principal_direction_collapse(vecs)
        # Centering sonrasi rank 3 -> 3 esit singular value -> ilk oran ~1/3
        assert math.isclose(result, 1 / 3, rel_tol=1e-4)

    def test_range(self):
        from humanai_detect.embeddings.anisotropy import principal_direction_collapse
        rng = np.random.default_rng(1)
        vecs = rng.standard_normal((30, 32))
        result = principal_direction_collapse(vecs)
        assert 0.0 <= result <= 1.0


class TestCosineNeighborhoodDispersion:
    def test_identical_vectors_high_dispersion(self):
        from humanai_detect.embeddings.anisotropy import cosine_neighborhood_dispersion
        vecs = np.ones((5, 4), dtype=float)
        result = cosine_neighborhood_dispersion(vecs, k=3)
        assert math.isclose(result, 1.0, rel_tol=1e-5)

    def test_single_vector_returns_zero(self):
        from humanai_detect.embeddings.anisotropy import cosine_neighborhood_dispersion
        result = cosine_neighborhood_dispersion(np.array([[1.0, 2.0]]), k=1)
        assert result == 0.0

    def test_returns_float(self):
        from humanai_detect.embeddings.anisotropy import cosine_neighborhood_dispersion
        rng = np.random.default_rng(2)
        vecs = rng.standard_normal((15, 16))
        result = cosine_neighborhood_dispersion(vecs, k=5)
        assert isinstance(result, float) and math.isfinite(result)


# ---------------------------------------------------------------------------
# _encoder.py — pooling mantigini mock tensors ile test et
# ---------------------------------------------------------------------------

class TestPooling:
    def _make_tensors(self, batch=2, seq=4, hidden=8):
        import torch
        hidden_t = torch.ones(batch, seq, hidden)
        mask = torch.ones(batch, seq, dtype=torch.long)
        return hidden_t, mask

    def test_cls_pooling_returns_first_token(self):
        import torch
        from humanai_detect.embeddings._encoder import _pool
        h, m = self._make_tensors()
        h[:, 0, :] = 99.0  # [CLS] tokeni farkli yap
        result = _pool(h, m, "cls")
        assert result.shape == (2, 8)
        assert torch.all(result == 99.0)

    def test_mean_pooling_shape(self):
        import torch
        from humanai_detect.embeddings._encoder import _pool
        h, m = self._make_tensors()
        result = _pool(h, m, "mean")
        assert result.shape == (2, 8)

    def test_max_pooling_shape(self):
        import torch
        from humanai_detect.embeddings._encoder import _pool
        h, m = self._make_tensors()
        result = _pool(h, m, "max")
        assert result.shape == (2, 8)

    def test_invalid_pooling_raises(self):
        import torch
        from humanai_detect.embeddings._encoder import _pool
        h, m = self._make_tensors()
        with pytest.raises(ValueError):
            _pool(h, m, "unknown")
