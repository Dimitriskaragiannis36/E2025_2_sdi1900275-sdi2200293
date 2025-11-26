import argparse
from pathlib import Path

from search.loader import load_model, load_index_files, load_query, load_dataset
from search.runner import run_nlsh_search

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data", required=True)
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("-i", "--index", required=True)
    parser.add_argument("-o", "--output", required=True)

    parser.add_argument("-type", required=True, choices=["mnist", "sift"])
    parser.add_argument("-N", type=int, default=1)
    parser.add_argument("-R", type=float, default=None)
    parser.add_argument("-T", type=int, default=5)
    parser.add_argument("-range", type=str, default="true")

    args = parser.parse_args()

    if args.R is None:
        args.R = 2000.0 if args.type == "mnist" else 2800.0

    index_dir = Path(args.index)

    print("=== Loading index files ===")
    meta, partitions, inverted = load_index_files(index_dir)

    print("=== Loading model ===")
    model = load_model(index_dir / "model.pth", meta)

    print("=== Loading dataset & query ===")
    X = load_dataset(args.data, args.type)
    Q = load_query(args.query, args.type)

    # ----------------------------------------------------------------------
    #δημιουργία φακέλου εξόδου
    # ----------------------------------------------------------------------
    base_output_dir = Path(args.output)
    final_out_dir = base_output_dir / args.type
    final_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {final_out_dir}")
    # ----------------------------------------------------------------------

    run_nlsh_search(model, X, Q, meta, partitions, inverted, args, final_out_dir)


if __name__ == "__main__":
    main()
