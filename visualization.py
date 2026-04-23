import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Sequence


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    medoid_indices: List[int],
    output_path: Optional[str] = None,
    title: str = "CLARANS Clustering",
    feature_names: Optional[Sequence[str]] = None,
    x_col: int = 0,
    y_col: int = 1,
):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[1] < 2:
        raise ValueError("plot_clusters_2d requires at least 2 dimensions.")
    if not (0 <= x_col < X.shape[1]) or not (0 <= y_col < X.shape[1]):
        raise ValueError("x_col and y_col must be valid column indices.")
    if x_col == y_col:
        raise ValueError("x_col and y_col must be different.")

    X_plot = X[:, [x_col, y_col]]
    medoids = X_plot[medoid_indices]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    x_label = feature_names[x_col] if x_col < len(feature_names) else f"Feature {x_col}"
    y_label = feature_names[y_col] if y_col < len(feature_names) else f"Feature {y_col}"

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_points = X_plot[labels == label]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {label}",
            alpha=0.75,
        )

    plt.scatter(
        medoids[:, 0],
        medoids[:, 1],
        c="black",
        marker="X",
        s=220,
        edgecolors="white",
        linewidths=1.5,
        label="Medoids",
    )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()
