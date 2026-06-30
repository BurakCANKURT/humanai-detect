"""Cosine uzaklik tabanli ozellik kumeleme — redundant ozellik tespiti."""

from __future__ import annotations

import numpy as np


def cosine_feature_clustering(features: np.ndarray, eps: float) -> np.ndarray:
    """Ozellikleri cosine uzakligina gore kumeler; her ozellige kume etiketi atar.

    features: [n_samples, n_features] — her SUTUN bir ozellik vektoru.
    eps     : iki ozellik arasindaki max cosine mesafesi ayni kumede sayilmak icin.

    Algoritma: precomputed cosine distance matrisi + agglomerative clustering
    (complete linkage — en buyuk ici kume mesafesini eps altinda tutar).

    Donus: [n_features] tamsayi kume etiketi dizisi.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances

    n_features = features.shape[1]
    if n_features < 2:
        return np.zeros(n_features, dtype=int)

    feature_vecs = features.T  # [n_features, n_samples]
    dist_matrix = cosine_distances(feature_vecs)  # [n_features, n_features]

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=eps,
        metric="precomputed",
        linkage="complete",
    )
    labels: np.ndarray = clustering.fit_predict(dist_matrix)
    return labels


def select_cluster_representatives(
    features: np.ndarray,
    cluster_labels: np.ndarray,
) -> np.ndarray:
    """Her kumeden en yuksek varyansli ozelligin indeksini secer.

    Donus: [n_clusters] secilen sutun indeksleri.
    """
    variances = features.var(axis=0)
    n_clusters = cluster_labels.max() + 1
    selected: list[int] = []
    for c in range(n_clusters):
        cluster_idx = np.where(cluster_labels == c)[0]
        best = cluster_idx[variances[cluster_idx].argmax()]
        selected.append(int(best))
    return np.array(sorted(selected))
