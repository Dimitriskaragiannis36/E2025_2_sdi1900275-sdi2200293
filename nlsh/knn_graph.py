import argparse
import os
import shutil
from nlsh_build import build_and_load_knn_graph
from graphs.kahip_wrapper import partition_knn_graph


def autodetect_exec():
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


def main():
    parser = argparse.ArgumentParser(description="Build and validate KNN graph")
    parser.add_argument("--dataset", "-d", choices=["mnist", "sift"], required=True)
    parser.add_argument("--k", "-k", type=int, default=10)

    parser.add_argument("--exec", default=autodetect_exec())
    parser.add_argument("--nblocks", type=int, default=8)
    parser.add_argument("--imbalance", type=float, default=0.03)

    args = parser.parse_args()

    if args.exec is None:
        print("ERROR: Could not locate 'search' executable.")
        exit(1)

    print(f"✔ Using ANN executable: {args.exec}")

    #βήμα 1: Κατασκευή KNN γράφου
    graph, csv_path = build_and_load_knn_graph(
        dataset_type=args.dataset,
        k=args.k,
        ann_exec=args.exec
    )

    print(" CSV produced at:", csv_path)
    print(" Nodes loaded:", len(graph))

    #Βήμα 2: Κατανομή γράφου με KaHIP
    print("Running KaHIP partitioning...")
    blocks, parts_map, meta = partition_knn_graph(
        graph,
        nblocks=args.nblocks,
        imbalance=args.imbalance,
        mode=1,
        seed=1,
        write_prefix=f"data/{args.dataset}/kahip"
    )

    print("✔ KaHIP completed!")
    print("Number of blocks:", len(blocks))


if __name__ == "__main__":
    main()
