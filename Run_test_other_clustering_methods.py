"""Run PCA + baseline clustering methods on all datasets.

This script mirrors the dataset configuration that is used for the
VaDE experiments and evaluates three classic clustering baselines:
KMeans, Louvain, and Leiden. Each dataset is projected with PCA before
clustering and the clustering quality is reported with ACC, NMI and ARI.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import networkx as nx
import numpy as np
from community import community_louvain
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from utility import leiden_clustering, set_random_seed


@dataclass
class DatasetSpec:
    """Holds file paths (or callables) for loading a dataset."""

    name: str
    data_loader: Callable[[], np.ndarray]
    label_loader: Callable[[], np.ndarray]

    def load(self) -> Dict[str, np.ndarray]:
        data = self.data_loader().astype(np.float64)
        labels = self.label_loader().astype(int)
        return {"data": data, "labels": labels}


def get_dataset_specs() -> List[DatasetSpec]:
    """Returns all dataset configurations to evaluate."""

    return [
        DatasetSpec(
            name="Algae",
            data_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/Algae/Algae_process.npy"
            ),
            label_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/Algae/Algae_label.npy"
            )[:, 0].astype(int),
        ),
        DatasetSpec(
            name="HP_15",
            data_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/HP/HP_X_processed.npy"
            ),
            label_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/HP/HP_Y_processed.npy"
            ).astype(int),
        ),
        DatasetSpec(
            name="NC_9",
            data_loader=lambda: np.flip(
                np.load(
                    "/mnt/sda/gene/zhangym/VADER/Data/NC_9/X_reference_9.npy"
                ),
                axis=1,
            ),
            label_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/NC_9/y_reference_9.npy"
            ).astype(int),
        ),
        DatasetSpec(
            name="NC_All",
            data_loader=lambda: np.flip(
                np.load(
                    "/mnt/sda/gene/zhangym/VADER/Data/NC_9/X_reference.npy"
                ),
                axis=1,
            ),
            label_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/NC_9/y_reference.npy"
            ).astype(int),
        ),
        DatasetSpec(
            name="Ocean_3",
            data_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/Ocean_3/Ocean_train_process.npy"
            ),
            label_loader=lambda: np.repeat([0, 1, 2], 50),
        ),
        DatasetSpec(
            name="Marine_7",
            data_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/Marine_7/Marine_7.npy"
            ),
            label_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/Marine_7/Marine_7_label.npy"
            ).astype(int),
        ),
        DatasetSpec(
            name="Neuron",
            data_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/Neuron/X_Neuron.npy"
            ),
            label_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/Neuron/Y_Neuron.npy"
            ).astype(int),
        ),
        DatasetSpec(
            name="Probiotics",
            data_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/Probiotics/X_probiotics.npy"
            ),
            label_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/Probiotics/Y_probiotics.npy"
            ).astype(int),
        ),
        DatasetSpec(
            name="MTB_Scientific",
            data_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/MTB_drug/MTB_Drug_scientific_X.npy"
            ),
            label_loader=lambda: np.load(
                "/mnt/sda/gene/zhangym/VADER/Data/MTB_drug/MTB_Drug_scientific_Y.npy"
            ).astype(int),
        ),
    ]


def prepare_features(data: np.ndarray, target_components: int, seed: int) -> np.ndarray:
    """Standardize the data and apply PCA."""
    pca = PCA(n_components=target_components, random_state=seed)
    return pca.fit_transform(data)


def kmeans_clustering(features: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """Run KMeans with a fixed number of clusters."""

    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    return model.fit_predict(features)


def louvain_clustering(
    features: np.ndarray, n_neighbors: int, resolution: float, seed: int
) -> np.ndarray:
    """Build a KNN graph and cluster with Louvain community detection."""

    nn_graph = NearestNeighbors(n_neighbors=n_neighbors)
    nn_graph.fit(features)
    adjacency = nn_graph.kneighbors_graph(features, mode="distance").tocoo()
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        zip(adjacency.row, adjacency.col, adjacency.data)
    )
    partition = community_louvain.best_partition(
        graph, resolution=resolution, random_state=seed
    )
    labels = np.array([partition[i] for i in range(features.shape[0])])
    return labels


def leiden_wrapper(
    features: np.ndarray, n_neighbors: int, resolution: float, seed: int
) -> np.ndarray:
    """Convenience wrapper around the existing Leiden helper."""

    return leiden_clustering(
        spectra=features, n_neighbors=n_neighbors, resolution=resolution, seed=seed
    )


def calculate_acc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Accuracy with Hungarian matching."""

    from scipy.optimize import linear_sum_assignment

    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size

    D = int(max(y_pred.max(), y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for yp, yt in zip(y_pred, y_true):
        w[yp, yt] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = ind[:, w[ind[0], ind[1]] > 0]
    if ind.size == 0:
        return 0.0
    return float(w[ind[0], ind[1]].sum()) / y_pred.size


def evaluate_clustering(
    labels_true: np.ndarray, labels_pred: np.ndarray
) -> Dict[str, float]:
    return {
        "acc": calculate_acc(labels_pred, labels_true),
        "nmi": normalized_mutual_info_score(labels_true, labels_pred),
        "ari": adjusted_rand_score(labels_true, labels_pred),
    }


def filter_specs(specs: Sequence[DatasetSpec], include: Iterable[str] | None) -> List[DatasetSpec]:
    if not include:
        return list(specs)
    include_set = {name.strip() for name in include}
    return [spec for spec in specs if spec.name in include_set]


def run_evaluation(args: argparse.Namespace) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    specs = filter_specs(get_dataset_specs(), args.datasets)

    for spec in specs:
        print(f"\n===== Dataset: {spec.name} =====")
        payload = spec.load()
        data, labels = payload["data"], payload["labels"]
        print(
            f"Samples: {data.shape[0]}, Features: {data.shape[1]}, Classes: {len(np.unique(labels))}"
        )

        features = prepare_features(data, args.pca_components, args.seed)
        n_clusters = len(np.unique(labels))

        # KMeans
        km_labels = kmeans_clustering(features, n_clusters=n_clusters, seed=args.seed)
        km_metrics = evaluate_clustering(labels, km_labels)
        print(
            f"KMeans -> ACC: {km_metrics['acc']:.4f}, NMI: {km_metrics['nmi']:.4f}, ARI: {km_metrics['ari']:.4f}"
        )
        results.append(row(spec.name, "kmeans", km_metrics))

        # Louvain
        lou_labels = louvain_clustering(
            features, n_neighbors=args.n_neighbors, resolution=args.resolution, seed=args.seed
        )
        lou_metrics = evaluate_clustering(labels, lou_labels)
        print(
            f"Louvain -> ACC: {lou_metrics['acc']:.4f}, NMI: {lou_metrics['nmi']:.4f}, ARI: {lou_metrics['ari']:.4f}"
        )
        results.append(row(spec.name, "louvain", lou_metrics))

        # Leiden
        lei_labels = leiden_wrapper(
            features, n_neighbors=args.n_neighbors, resolution=args.resolution, seed=args.seed
        )
        lei_metrics = evaluate_clustering(labels, lei_labels)
        print(
            f"Leiden -> ACC: {lei_metrics['acc']:.4f}, NMI: {lei_metrics['nmi']:.4f}, ARI: {lei_metrics['ari']:.4f}"
        )
        results.append(row(spec.name, "leiden", lei_metrics))

    return results


def row(dataset: str, method: str, metrics: Dict[str, float]) -> Dict[str, str]:
    return {
        "dataset": dataset,
        "method": method,
        "acc": f"{metrics['acc']:.4f}",
        "nmi": f"{metrics['nmi']:.4f}",
        "ari": f"{metrics['ari']:.4f}",
    }


def save_results(path: str, rows: List[Dict[str, str]]) -> None:
    header = ["dataset", "method", "acc", "nmi", "ari"]
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA + clustering baselines (KMeans/Louvain/Leiden) on all datasets."
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components before clustering (default: 50)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="Neighbors used when building the KNN graph for graph-based clustering.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution parameter for Louvain/Leiden.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for deterministic behavior.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        help="Optional subset of dataset names to evaluate (default: all).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Optional path to store the aggregated metrics as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    rows = run_evaluation(args)
    if args.output_csv:
        save_results(args.output_csv, rows)


if __name__ == "__main__":
    main()
