#!/usr/bin/env python3
import argparse
import os
import shutil
from nlsh_build import build_and_load_knn_graph

def autodetect_exec():
    """
    Εντοπίζει το 'search' executable οπουδήποτε μέσα στο $HOME,
    συν τοπικούς φακέλους και το PATH.
    Χωρίς κανέναν περιορισμό βάθους.
    """

    candidates = [
        "./bin/search",
        "bin/search",
        "../bin/search",
    ]

    # 1. Γρήγορο check για τα πιο συνηθισμένα σημεία
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    # 2. Ψάχνει ΟΛΟ το $HOME (χωρίς limit)
    home = os.path.expanduser("~")
    for root, dirs, files in os.walk(home):
        if "search" in files:
            return os.path.join(root, "search")

    # 3. PATH fallback
    import shutil
    found = shutil.which("search")
    if found:
        return found

    return None

def main():
    parser = argparse.ArgumentParser(description="Build and validate KNN graph")
    parser.add_argument(
        "--dataset", "-d",
        choices=["mnist", "sift"],
        required=True,
        help="Dataset type (mnist or sift)"
    )
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=10,
        help="Number of neighbors (default: 10)"
    )

    default_exec = autodetect_exec()

    parser.add_argument(
        "--exec",
        default=default_exec,
        help="Path to the ANN executable from Assignment 1 (auto-detected if possible)"
    )

    args = parser.parse_args()

    if args.exec is None:
        print("❌ ERROR: Could not locate the 'search' executable!")
        print("➡ Please provide it manually with:  --exec PATH_TO_SEARCH")
        exit(1)

    print(f"✔ Using ANN executable: {args.exec}")

    # --- Run ANN search and load CSV ---
    graph, csv_path = build_and_load_knn_graph(
        dataset_type=args.dataset,
        k=args.k,
        ann_exec=args.exec
    )

    print("📥 CSV produced at:", csv_path)
    print("📌 Nodes loaded:", len(graph))

    # --- Show first 5 entries ---
    print("\n🔍 First 5 entries:")
    for i, (node, neighbors) in enumerate(graph.items()):
        print(f"Node {node} -> {neighbors}")
        if i >= 4:
            break

    # --- Check consistency ---
    expected_k = None
    for node, neighbors in graph.items():
        if expected_k is None:
            expected_k = len(neighbors)
            print(f"\n➡ Expected k = {expected_k}")
        elif len(neighbors) != expected_k:
            print(f"❌ Node {node} has {len(neighbors)} neighbors (expected {expected_k})")
            break
    else:
        print("✅ All nodes have correct number of neighbors")

    # --- Check neighbor ID range ---
    max_node = max(graph.keys())
    for node, neighbors in graph.items():
        for nb in neighbors:
            if nb < 0 or nb > max_node:
                print(f"❌ Invalid neighbor ID {nb} for node {node}")
                exit(1)

    print("✅ All neighbor IDs are within valid range")

    print(f"\n🎉 {args.dataset.upper()} KNN graph built, saved, and verified successfully!")


if __name__ == "__main__":
    main()
