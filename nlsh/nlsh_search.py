import argparse
from pathlib import Path
import numpy as np
import torch

from search.loader import load_model, load_index_files, load_query, load_dataset
from search.predictor import predict_bins


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data", required=True, help="input.dat")
    parser.add_argument("-q", "--query", required=True, help="query.dat")
    parser.add_argument("-i", "--index", required=True, help="index directory")
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

    print("Running STEP 1: prediction for query")
    all_probs = []
    for idx, q in enumerate(Q):
        probs = predict_bins(model, q)  #σχέδιο [nblocks,]
        all_probs.append(probs.numpy())

    all_probs = np.vstack(all_probs)  #σχέδιο [num_queries, nblocks]

    print("Probabilities:")
    print(all_probs[0])
    
    output_path = Path(args.output)
    np.savetxt(output_path, all_probs, fmt="%.6f")
    print(f"✓ All probabilities saved to {output_path}")
    print("\n✓ Step 1 complete.")


if __name__ == "__main__":
    main()
