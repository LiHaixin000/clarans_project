import csv
from typing import Dict, List, Optional

from clarans import Clarans


def run_experiments(
    X,
    k_values: List[int],
    numlocal_values: List[int],
    maxneighbor_values: List[int],
    distance: str = "euclidean",
    use_distance_matrix: bool = False,
    seeds: Optional[List[int]] = None,
):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    results = []

    for k in k_values:
        for numlocal in numlocal_values:
            for maxneighbor in maxneighbor_values:
                for seed in seeds:
                    model = Clarans(
                        n_clusters=k,
                        numlocal=numlocal,
                        maxneighbor=maxneighbor,
                        distance=distance,
                        use_distance_matrix=use_distance_matrix,
                        random_state=seed,
                    )
                    result = model.fit(X)

                    results.append(
                        {
                            "k": k,
                            "numlocal": numlocal,
                            "maxneighbor": maxneighbor,
                            "seed": seed,
                            "cost": result.cost,
                            "runtime": result.runtime,
                            "medoids": result.medoid_indices,
                            "local_costs": result.local_costs,
                        }
                    )

    return results


def save_results_csv(results: List[Dict], output_file: str):
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
