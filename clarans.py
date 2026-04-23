import numpy as np
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ClaransResult:
    medoid_indices: List[int]
    labels: np.ndarray
    cost: float
    runtime: float
    local_costs: List[float]


class Clarans:
    def __init__(
        self,
        n_clusters: int,
        numlocal: int = 2,
        maxneighbor: int = 250,
        distance: str = "euclidean",
        use_distance_matrix: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.numlocal = numlocal
        self.maxneighbor = maxneighbor
        self.distance = distance.lower()
        self.use_distance_matrix = use_distance_matrix
        self.random_state = random_state

        if self.distance not in {"euclidean", "manhattan"}:
            raise ValueError("distance must be 'euclidean' or 'manhattan'")

        self.data = None
        self.distance_matrix = None

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def _pairwise_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.distance == "euclidean":
            return float(np.linalg.norm(a - b))
        return float(np.sum(np.abs(a - b)))

    def _build_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = self._pairwise_distance(X[i], X[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def _distance_idx(self, i: int, j: int) -> float:
        if self.use_distance_matrix and self.distance_matrix is not None:
            return float(self.distance_matrix[i, j])
        return self._pairwise_distance(self.data[i], self.data[j])

    def _initialize_random_medoids(self, n: int) -> List[int]:
        return random.sample(range(n), self.n_clusters)

    def _assign_clusters(self, medoids: List[int]) -> Tuple[np.ndarray, float]:
        n = len(self.data)
        labels = np.empty(n, dtype=int)
        total_cost = 0.0

        for i in range(n):
            best_cluster = -1
            best_dist = float("inf")
            for cluster_id, medoid_idx in enumerate(medoids):
                d = self._distance_idx(i, medoid_idx)
                if d < best_dist:
                    best_dist = d
                    best_cluster = cluster_id
            labels[i] = best_cluster
            total_cost += best_dist

        return labels, total_cost

    def _compute_cost(self, medoids: List[int]) -> float:
        _, cost = self._assign_clusters(medoids)
        return cost

    def _generate_random_neighbor(self, current_medoids: List[int], n: int) -> List[int]:
        medoid_set = set(current_medoids)
        medoid_to_replace = random.choice(current_medoids)

        non_medoids = [idx for idx in range(n) if idx not in medoid_set]
        replacement = random.choice(non_medoids)

        neighbor = current_medoids.copy()
        replace_pos = neighbor.index(medoid_to_replace)
        neighbor[replace_pos] = replacement
        return neighbor

    def _local_search(self, n: int) -> Tuple[List[int], float]:
        current = self._initialize_random_medoids(n)
        current_cost = self._compute_cost(current)

        j = 1
        while j <= self.maxneighbor:
            neighbor = self._generate_random_neighbor(current, n)
            neighbor_cost = self._compute_cost(neighbor)

            if neighbor_cost < current_cost:
                current = neighbor
                current_cost = neighbor_cost
                j = 1
            else:
                j += 1

        return current, current_cost

    def fit(self, X: np.ndarray) -> ClaransResult:
        start = time.perf_counter()

        X = np.asarray(X, dtype=float)
        n = len(X)

        if n == 0:
            raise ValueError("Empty dataset.")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        if self.n_clusters > n:
            raise ValueError("n_clusters cannot exceed number of objects.")

        self.data = X

        if self.use_distance_matrix:
            self.distance_matrix = self._build_distance_matrix(X)

        best_medoids = None
        best_cost = float("inf")
        local_costs = []

        for _ in range(self.numlocal):
            medoids, cost = self._local_search(n)
            local_costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_medoids = medoids

        labels, final_cost = self._assign_clusters(best_medoids)
        runtime = time.perf_counter() - start

        return ClaransResult(
            medoid_indices=best_medoids,
            labels=labels,
            cost=final_cost,
            runtime=runtime,
            local_costs=local_costs,
        )
