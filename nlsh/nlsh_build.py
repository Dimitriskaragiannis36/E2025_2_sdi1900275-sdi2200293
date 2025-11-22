import argparse
from io_utils.exec_finder import autodetect_exec
from pipeline.builder import build_pipeline

def main():

    parser = argparse.ArgumentParser(description="NLSH build pipeline")
    parser.add_argument("--dataset", required=True, choices=["mnist", "sift"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--method", type=str, default="ivfflat")
    parser.add_argument("--ann_exec", type=str, default=autodetect_exec())
    parser.add_argument("--nblocks", type=int, default=8)
    parser.add_argument("--imbalance", type=float, default=0.03)

    args = parser.parse_args()

    if args.ann_exec is None:
        print("ERROR: Could not locate 'search' executable.")
        exit(1)

    print(f"✔ Using ANN executable: {args.ann_exec}")

    build_pipeline(
        dataset_type=args.dataset,
        k=args.k,
        method=args.method,
        ann_exec=args.ann_exec,
        nblocks=args.nblocks,
        imbalance=args.imbalance,
    )

if __name__ == "__main__":
    main()
