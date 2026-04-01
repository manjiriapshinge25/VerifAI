"""
src/clustering/narrative_clusterer.py
---------------------------------------
Unsupervised narrative clustering using UMAP + HDBSCAN.
This is the UML phase of VerifAI.
TODO: Try different UMAP metrics (cosine, manhattan) and HDBSCAN cluster sizes.
"""

import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, normalized_mutual_info_score


class NarrativeClusterer:
    """
    Reduces CLIP embeddings with UMAP and clusters with HDBSCAN.
    Discovers latent misinformation narratives without any labels.
    """

    def __init__(self, config):
        self.umap_n_components = config["clustering"]["umap_n_components"]
        self.umap_n_neighbors = config["clustering"]["umap_n_neighbors"]
        self.min_cluster_size = config["clustering"]["min_cluster_size"]
        self.min_samples = config["clustering"]["min_samples"]

        self.reducer = None
        self.clusterer = None
        self.reduced_embs = None
        self.cluster_labels = None

    def fit(self, embeddings):
        """
        Fit UMAP reducer and HDBSCAN clusterer on embeddings.

        Args:
            embeddings: np.array of shape (N, D)
        Returns:
            cluster_labels: np.array of shape (N,), -1 = noise
        """
        print("[NarrativeClusterer] Reducing dimensions with UMAP...")
        self.reducer = umap.UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            metric="cosine",
            random_state=42,
        )
        self.reduced_embs = self.reducer.fit_transform(embeddings)

        print("[NarrativeClusterer] Clustering with HDBSCAN...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            prediction_data=True,
        )
        self.cluster_labels = self.clusterer.fit_predict(self.reduced_embs)

        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        noise_ratio = (self.cluster_labels == -1).mean()
        print(f"[NarrativeClusterer] Found {n_clusters} clusters | Noise: {noise_ratio:.1%}")

        return self.cluster_labels

    def evaluate(self, true_labels=None):
        """
        Compute clustering quality metrics.
        TODO: Add Davies-Bouldin Index for additional evaluation.
        """
        metrics = {}

        # Filter out noise points for silhouette
        mask = self.cluster_labels != -1
        if mask.sum() > 1:
            metrics["silhouette_score"] = silhouette_score(
                self.reduced_embs[mask], self.cluster_labels[mask]
            )

        if true_labels is not None:
            metrics["nmi"] = normalized_mutual_info_score(true_labels, self.cluster_labels)

        print(f"[NarrativeClusterer] Metrics: {metrics}")
        return metrics

    def visualize(self, labels=None, save_path="results/clusters_umap.png"):
        """
        2D UMAP visualization of clusters.
        TODO: Add interactive Plotly version for the Streamlit dashboard.
        """
        viz_reducer = umap.UMAP(n_components=2, random_state=42)
        coords_2d = viz_reducer.fit_transform(self.reduced_embs)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=self.cluster_labels,
            cmap="Spectral",
            s=5,
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Cluster ID (-1 = noise)")
        plt.title("VerifAI — Narrative Clusters (UMAP + HDBSCAN)")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[NarrativeClusterer] Saved cluster plot → {save_path}")
