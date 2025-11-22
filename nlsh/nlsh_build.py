import argparse
import os
import shutil
from pathlib import Path

from graphs.knn_builder import build_and_load_knn_graph


def autodetect_exec():
    """
    Βρίσκει το search executable αυτόματα.
    """
    candidates = [
        "./bin/search",
        "bin/search",
        "../bin/search",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    home = os.path.expanduser("~")
    for root, dirs, files in os.walk(home):
        if "search" in files:
            return os.path.join(root, "search")

    found = shutil.which("search")
    if found:
        return found

    return None


def build_pipeline(dataset_type, k, method, ann_exec):

    print(f"=== [1] Building kNN graph for dataset: {dataset_type} ===")
    graph, csv_path = build_and_load_knn_graph(
        dataset_type=dataset_type,
        k=k,
        method=method,
        ann_exec=ann_exec
    )

    print("✓ KNN graph built:", csv_path)
    print("✓ Graph size:", len(graph))

    print("=== Pipeline finished ===")
    return graph


def main():

    parser = argparse.ArgumentParser(description="NLSH build pipeline")
    parser.add_argument("--dataset", required=True, choices=["mnist", "sift"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--method", type=str, default="ivfflat")

    #αυτόματη ανίχνευση του εκτελέσιμου αρχείου
    parser.add_argument("--ann_exec", type=str, default=autodetect_exec())

    args = parser.parse_args()

    #έλεγχος αν βρέθηκε το εκτελέσιμο αρχείο
    if args.ann_exec is None:
        print("ERROR: Could not locate 'search' executable.")
        exit(1)

    print(f"✔ Using ANN executable: {args.ann_exec}")

    build_pipeline(
        dataset_type=args.dataset,
        k=args.k,
        method=args.method,
        ann_exec=args.ann_exec
    )


if __name__ == "__main__":
    main()
