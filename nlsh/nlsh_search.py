import argparse
from pathlib import Path
import numpy as np

from search.loader import load_model, load_index_files, load_query, load_dataset
from search.nlsh_steps import compute_probabilities, select_top_T_bins


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

    # ----------------------------------------------------------------------
    #δημιουργία φακέλου εξόδου
    # ----------------------------------------------------------------------
    base_output_dir = Path(args.output)
    final_out_dir = base_output_dir / args.type
    final_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {final_out_dir}")
    # ----------------------------------------------------------------------

    print("Running STEP 1: prediction for query")
  
    all_probs = compute_probabilities(model, Q)

    #αποθήκευση αποτελεσμάτων βήματος 1
    step1_file = final_out_dir / "probs.txt"
    np.savetxt(step1_file, all_probs, fmt="%.6f")
    print(f"✓ All probabilities saved to {step1_file}")

    print("\nRunning STEP 2: Multi-Probe bin selection")

    bins, probs = select_top_T_bins(all_probs, args.T)

    np.savetxt(final_out_dir / "bins.txt", bins, fmt="%d")
    np.savetxt(final_out_dir / "bins_probs.txt", probs, fmt="%.6f")

    print("✓ Step 2 complete.")



if __name__ == "__main__":
    main()
