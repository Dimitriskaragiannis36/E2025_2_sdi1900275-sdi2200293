from pathlib import Path
import numpy as np
from search.nlsh_steps import (
    compute_probabilities,
    select_top_T_bins,
    collect_candidates,
    exact_search
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

    #βήμα 4
    print("\nRunning STEP 4: Exact Search")

    range_mode = (args.range.lower() == "true")

    results = exact_search(
        X,
        Q,
        candidates,
        R=args.R,
        N=args.N,
        range_mode=range_mode
    )

    #αποθήκευση αποτελεσμάτων
    out_file = outdir / "results.txt"
    with open(out_file, "w") as f:
        for qi, res in enumerate(results):
            line = f"Query {qi}: "
            if range_mode:
                #μορφή: id:dist,...
                line += ",".join(f"{pid}:{dist:.4f}" for pid, dist in res)
            else:
                line += ",".join(f"{pid}:{dist:.4f}" for pid, dist in res)
            f.write(line + "\n")

    print(f"✓ Step 4 complete. Results saved to {out_file}")

    return {
        "probs": all_probs,
        "bins": bins,
        "candidates": candidates
    }
