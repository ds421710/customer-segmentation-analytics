"""
Phase 4: RFM Analysis & K-Means Customer Segmentation Module
Computes Recency, Frequency, Monetary metrics, deterministic RFM quintile
CRM segments, log-scaled K-Means clustering with Elbow/Silhouette evaluation,
and programmatic cluster-label derivation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

sns.set_theme(style="whitegrid")


def _fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> Tuple[np.ndarray, float]:
    """Fit K-Means using scikit-learn when available, or NumPy K-Means++ fallback."""
    if HAS_SKLEARN:
        km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        return labels, float(km.inertia_)

    rng = np.random.default_rng(random_state)
    best_labels = None
    best_inertia = float("inf")
    for _ in range(10):
        # K-Means++ initialization
        centroids = [X[rng.integers(0, len(X))]]
        for _ in range(1, n_clusters):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dists / (dists.sum() + 1e-12)
            centroids.append(X[rng.choice(len(X), p=probs)])
        centers = np.array(centroids)

        for _ in range(100):
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centers = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                for k in range(n_clusters)
            ])
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        inertia = float(sum(np.sum((X[labels == k] - centers[k]) ** 2) for k in range(n_clusters)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    return best_labels, best_inertia


def _compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute Silhouette Score using scikit-learn or exact NumPy pairwise distances."""
    if len(set(labels)) <= 1:
        return 0.0
    if HAS_SKLEARN:
        return float(silhouette_score(X, labels))

    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    scores = []
    for i in range(len(X)):
        same_mask = (labels == labels[i])
        same_mask[i] = False
        a_i = dists[i, same_mask].mean() if np.any(same_mask) else 0.0
        b_i = min(
            dists[i, labels == k].mean()
            for k in set(labels)
            if k != labels[i]
        )
        scores.append((b_i - a_i) / max(a_i, b_i, 1e-12))
    return float(np.mean(scores))


def compute_rfm_table(
    df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Compute Recency (days), Frequency (orders), and Monetary (£) per customer."""
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID", as_index=False)
        .agg(
            LastPurchaseDate=("InvoiceDate", "max"),
            FirstPurchaseDate=("InvoiceDate", "min"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        )
    )
    rfm["Recency"] = (snapshot_date - rfm["LastPurchaseDate"]).dt.days
    rfm["TenureDays"] = (snapshot_date - rfm["FirstPurchaseDate"]).dt.days
    rfm["Monetary"] = rfm["Monetary"].round(2)

    # Deterministic RFM quintile scoring (1 = lowest, 5 = highest)
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first", ascending=False), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first", ascending=True), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first", ascending=True), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    rfm["CRM_Segment"] = rfm.apply(_assign_crm_segment, axis=1)
    return rfm


def _assign_crm_segment(row: pd.Series) -> str:
    """Derive 8 standard CRM lifecycle segments deterministically from R, F, M scores."""
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
    fm = (f + m) / 2.0

    if r >= 4 and fm >= 4.0:
        return "Champions"
    if r >= 3 and fm >= 3.5:
        return "Loyal Customers"
    if r >= 4 and 2.0 <= fm < 4.0:
        return "Potential Loyalists"
    if r >= 4 and fm < 2.0:
        return "New Customers"
    if r <= 2 and fm >= 3.5:
        return "At-Risk Customers"
    if r == 3 and fm < 3.5:
        return "Customers Needing Attention"
    if r == 2 and fm < 3.5:
        return "Hibernating Customers"
    return "Lost Customers"


def _scale_rfm_features(rfm_df: pd.DataFrame) -> np.ndarray:
    """Apply log1p transform (to handle wholesale skew) and z-score standardization."""
    features = rfm_df[["Recency", "Frequency", "Monetary"]].copy()
    features["Recency"] = np.log1p(features["Recency"].clip(lower=0))
    features["Frequency"] = np.log1p(features["Frequency"].clip(lower=0))
    features["Monetary"] = np.log1p(features["Monetary"].clip(lower=0))
    arr = features.to_numpy(dtype=float)
    if HAS_SKLEARN:
        return StandardScaler().fit_transform(arr)
    return (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-9)


def evaluate_kmeans_k(
    rfm_df: pd.DataFrame, k_range: Iterable[int] = range(2, 8)
) -> pd.DataFrame:
    """Compute Inertia (Elbow) and Silhouette Score across candidate k values."""
    X = _scale_rfm_features(rfm_df)
    results = []
    n_samples = len(X)
    for k in k_range:
        if k >= n_samples:
            continue
        labels, inertia = _fit_kmeans(X, n_clusters=k, random_state=42)
        sil = _compute_silhouette(X, labels)
        results.append({"k": k, "Inertia": round(inertia, 2), "SilhouetteScore": round(sil, 4)})
    return pd.DataFrame(results)


def run_kmeans_segmentation(
    rfm_df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit K-Means on log-scaled RFM features and programmatically derive
    business labels from each cluster's mean R, F, and M values.
    """
    out = rfm_df.copy()
    X = _scale_rfm_features(out)
    labels, _ = _fit_kmeans(X, n_clusters=n_clusters, random_state=random_state)
    out["Cluster"] = labels

    profile = (
        out.groupby("Cluster", as_index=False)
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            TotalRevenue=("Monetary", "sum"),
        )
    )
    total_cust = profile["Customers"].sum()
    total_rev = profile["TotalRevenue"].sum()
    profile["CustomerSharePct"] = (profile["Customers"] / total_cust * 100).round(2)
    profile["RevenueSharePct"] = (profile["TotalRevenue"] / total_rev * 100).round(2)

    # Programmatic ranking: higher Frequency & Monetary, lower Recency => higher value
    z_f = (profile["AvgFrequency"] - profile["AvgFrequency"].mean()) / (profile["AvgFrequency"].std() + 1e-9)
    z_m = (profile["AvgMonetary"] - profile["AvgMonetary"].mean()) / (profile["AvgMonetary"].std() + 1e-9)
    z_r = (profile["AvgRecency"] - profile["AvgRecency"].mean()) / (profile["AvgRecency"].std() + 1e-9)
    profile["ValueIndex"] = (z_f + z_m - z_r).round(3)
    profile = profile.sort_values("ValueIndex", ascending=False).reset_index(drop=True)

    label_pool = [
        "Champions / High-Value",
        "Loyal / Steady Buyers",
        "Potential / Developing",
        "At-Risk / Hibernating",
        "Occasional / Low-Spend",
        "Lost / Churned",
    ]
    cluster_to_label: Dict[int, str] = {}
    for idx, row in profile.iterrows():
        c_id = int(row["Cluster"])
        if idx == 0:
            cluster_to_label[c_id] = "Champions / High-Value"
        elif idx == len(profile) - 1:
            cluster_to_label[c_id] = "Hibernating / Lost"
        elif row["AvgRecency"] > profile["AvgRecency"].median():
            cluster_to_label[c_id] = "At-Risk / Needs Attention"
        else:
            cluster_to_label[c_id] = label_pool[min(idx, len(label_pool) - 1)]

    profile["ClusterLabel"] = profile["Cluster"].map(cluster_to_label)
    out["ClusterLabel"] = out["Cluster"].map(cluster_to_label)

    for col in ["AvgRecency", "AvgFrequency", "AvgMonetary", "TotalRevenue"]:
        profile[col] = profile[col].round(2)

    return out, profile


def generate_rfm_charts(
    rfm_df: pd.DataFrame, k_eval_df: pd.DataFrame, output_dir: str | Path
) -> Dict[str, Path]:
    """Generate Elbow/Silhouette diagnostics and Cluster RFM visualizations."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.lineplot(data=k_eval_df, x="k", y="Inertia", marker="o", ax=axes[0], color="#1f77b4")
    axes[0].set_title("Elbow Method (Inertia vs. k)", fontweight="bold")
    sns.lineplot(data=k_eval_df, x="k", y="SilhouetteScore", marker="s", ax=axes[1], color="#2ca02c")
    axes[1].set_title("Silhouette Score vs. k", fontweight="bold")
    plt.tight_layout()
    p1 = out / "rfm_elbow_silhouette.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths["elbow_silhouette"] = p1

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=rfm_df,
        x="Recency",
        y="Monetary",
        hue="ClusterLabel",
        size="Frequency",
        sizes=(40, 250),
        alpha=0.8,
        ax=ax,
    )
    ax.set_title("Customer Segments: Recency vs. Monetary Spend", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recency (Days Since Last Purchase)")
    ax.set_ylabel("Total Monetary Spend (£)")
    plt.tight_layout()
    p2 = out / "rfm_clusters.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths["rfm_clusters"] = p2

    return paths
