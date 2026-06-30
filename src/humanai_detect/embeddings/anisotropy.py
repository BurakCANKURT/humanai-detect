"""Embedding uzay kalitesi metrikleri: anisotropy, principal-direction collapse, cosine-dispersion.

Bu metrikler modelin urettigi embedding'lerin ne kadar "tasınabilir" (transfer edilebilir)
ve ayırt edici oldugunu olcer. Yuksek anisotropy veya collapse -> embedding uzayi cokmis
demektir (tum vektorler benzer yonde), bu da siniflandirma performansini dusurebilir.
"""

from __future__ import annotations

import numpy as np


def compute_anisotropy(embeddings: np.ndarray) -> float:
    """Ortalama cift-yonlu cosine benzerligi tabanli anisotropy hesaplar.

    Deger araligi [-1, 1]:
      ~0  -> izotropik (ideal durum, vektorler tum yonlere esit dagilmis)
      ~1  -> anizotropik (vektorler tek bir yonde yigismis — dil modellerinde yaygin sorun)

    Kaynak: Ethayarajh (2019) "How contextual are contextualized word representations?"
    """
    if embeddings.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms  # [N, D]
    # Tum cift cosine benzerliklerinin ortalamasi = ortalama nokta carpimi
    sim_matrix = normed @ normed.T  # [N, N]
    n = sim_matrix.shape[0]
    # Diagonal (oz-benzerlik = 1) haric ortalama
    mask = ~np.eye(n, dtype=bool)
    return float(sim_matrix[mask].mean())


def principal_direction_collapse(embeddings: np.ndarray) -> float:
    """Birinci temel bilesene dusen varyans orani (PCA tabanli).

    0 -> varyans tum yonlere esit dagilmis (saglıklı)
    1 -> tum varyans tek bir yonde (tam collapse)
    """
    if embeddings.shape[0] < 2:
        return 0.0
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    variance = s ** 2
    total = variance.sum()
    if total == 0:
        return 1.0  # Sifir varyans -> tum vektorler ayni -> tam collapse
    return float(variance[0] / total)


def cosine_neighborhood_dispersion(embeddings: np.ndarray, k: int = 10) -> float:
    """k en yakin komsu cosine benzerliklerinin ortalamasi (dagilim olcusu).

    Dusuk deger -> vektorler birbirinden uzak (iyi ayirt edicilik).
    Yuksek deger -> vektorler yigismis (zayif ayirt edicilik).
    """
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    k = min(k, n - 1)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T  # [N, N]
    np.fill_diagonal(sim_matrix, -2.0)  # kendini komsu olarak secmesin

    dispersions: list[float] = []
    for i in range(n):
        top_k_idx = np.argpartition(sim_matrix[i], -k)[-k:]
        dispersions.append(float(sim_matrix[i, top_k_idx].mean()))

    return float(np.mean(dispersions))
