from pathlib import Path
import numpy as np
from search.nlsh_steps import (
    compute_probabilities,
    select_top_T_bins,
    collect_candidates
)


def run_nlsh_search(model, X, Q, meta, partitions, inverted, args, outdir: Path):

    #βήμα 1
    print("Running STEP 1: prediction for query")
    all_probs = compute_probabilities(model, Q)
    np.savetxt(outdir / "probs.txt", all_probs, fmt="%.6f")
    print("✓ Step 1 complete.")

    #βήμα 2
    print("\nRunning STEP 2: Multi-Probe bin selection")
    bins, probs = select_top_T_bins(all_probs, args.T)
    np.savetxt(outdir / "bins.txt", bins, fmt="%d")
    np.savetxt(outdir / "bins_probs.txt", probs, fmt="%.6f")
    print("✓ Step 2 complete.")

    #βήμα 3
    print("\nRunning STEP 3: Collecting candidates")
    candidates = collect_candidates(bins, inverted)
    with open(outdir / "candidates.txt", "w") as f:
        for i, cand in enumerate(candidates):
            f.write(f"Query {i}: {','.join(map(str, cand))}\n")
    print("✓ Step 3 complete.")

    return {
        "probs": all_probs,
        "bins": bins,
        "candidates": candidates
    }
