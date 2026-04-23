import argparse
import numpy as np

from clarans import Clarans
from data_utils import load_data
from experiment_runner import run_experiments, save_results_csv
from visualization import plot_clusters_2d


def parse_columns(columns_str):
    if columns_str is None or columns_str.strip() == "":
        return None
    return [c.strip() for c in columns_str.split(",")]


def main():
    parser = argparse.ArgumentParser(description="CLARANS clustering from scratch")

    parser.add_argument("--input", type=str, required=True, help="Path to input CSV/TXT file")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters")
    parser.add_argument("--numlocal", type=int, default=2, help="Number of local minima")
    parser.add_argument("--maxneighbor", type=int, default=250, help="Max examined neighbors")
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan"],
        help="Distance metric",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Comma-separated list of numeric columns to use",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply min-max normalization",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Apply z-score standardization",
    )
    parser.add_argument(
        "--use-distance-matrix",
        action="store_true",
        help="Precompute pairwise distance matrix",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot clusters in 2D",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Save plot to file instead of displaying",
    )
    parser.add_argument(
        "--plot-x",
        type=int,
        default=0,
        help="Column index for x-axis in plot",
    )
    parser.add_argument(
        "--plot-y",
        type=int,
        default=1,
        help="Column index for y-axis in plot",
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run a parameter grid instead of a single clustering",
    )
    parser.add_argument(
        "--experiment-output",
        type=str,
        default="experiment_results.csv",
        help="CSV file for experiment output",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Specify that the input file has no header row (use this for iris.csv)",
    )

    args = parser.parse_args()

    columns = parse_columns(args.columns)
    header = None if args.no_header else "infer"

    X, df = load_data(
        args.input,
        columns=columns,
        normalize=args.normalize,
        standardize=args.standardize,
        header=header,
    )

    if args.run_experiments:
        k_values = [2, 3, 4, 5, args.k]
        k_values = sorted(set([k for k in k_values if k <= len(X)]))
        numlocal_values = [1, 2, 3, 5]
        maxneighbor_values = [50, 100, 250, 500]
        seeds = [0, 1, 2, 3, 4]

        results = run_experiments(
            X,
            k_values=k_values,
            numlocal_values=numlocal_values,
            maxneighbor_values=maxneighbor_values,
            distance=args.distance,
            use_distance_matrix=args.use_distance_matrix,
            seeds=seeds,
        )
        save_results_csv(results, args.experiment_output)
        print(f"Saved experiment results to: {args.experiment_output}")
        return

    model = Clarans(
        n_clusters=args.k,
        numlocal=args.numlocal,
        maxneighbor=args.maxneighbor,
        distance=args.distance,
        use_distance_matrix=args.use_distance_matrix,
        random_state=args.seed,
    )

    result = model.fit(X)

    print("=== CLARANS RESULT ===")
    print(f"Number of objects: {len(X)}")
    print(f"Number of clusters (k): {args.k}")
    print(f"numlocal: {args.numlocal}")
    print(f"maxneighbor: {args.maxneighbor}")
    print(f"Distance: {args.distance}")
    print(f"Use distance matrix: {args.use_distance_matrix}")
    print(f"Medoid indices: {result.medoid_indices}")
    print("Medoid coordinates:")
    print(np.array(X)[result.medoid_indices])
    print(f"Final cost: {result.cost:.6f}")
    print(f"Runtime: {result.runtime:.6f} seconds")
    print(f"Local search costs: {result.local_costs}")

    if args.plot:
        feature_names = list(df.columns)
        plot_clusters_2d(
            X,
            result.labels,
            result.medoid_indices,
            output_path=args.plot_output,
            title="CLARANS Clustering",
            feature_names=feature_names,
            x_col=args.plot_x,
            y_col=args.plot_y,
        )
        if args.plot_output:
            print(f"Plot saved to: {args.plot_output}")


if __name__ == "__main__":
    main()
