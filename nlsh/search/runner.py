import numpy as np #για αποθήκευση αρχείων κειμένου

from search.output_writer import write_output_file #για την τελική αποθήκευση αποτελεσμάτων
from search.nlsh_steps import (
    compute_probabilities,
    select_top_T_bins,
    collect_candidates,
    exact_search
) #για τα βήματα της αναζήτησης

#εκτέλεση της διαδικασίας αναζήτησης Neural LSH
def run_nlsh_search(model, X, Q, inverted, args, outdir):

    #βήμα 1
    print("Running STEP 1: prediction for query")
    all_probs = compute_probabilities(model, Q) #πρόβλεψη πιθανοτήτων για όλα τα ερωτήματα
    np.savetxt(outdir / "probs.txt", all_probs, fmt="%.6f") #αποθήκευση πιθανοτήτων
    print("✓ Step 1 complete.") 

    #βήμα 2
    print("\nRunning STEP 2: Multi-Probe bin selection")
    bins, probs = select_top_T_bins(all_probs, args.T) #επιλογή των κορυφαίων Τ bins για κάθε ερώτημα
    np.savetxt(outdir / "bins.txt", bins, fmt="%d") #αποθήκευση επιλεγμένων bins
    np.savetxt(outdir / "bins_probs.txt", probs, fmt="%.6f") #αποθήκευση πιθανοτήτων επιλεγμένων bins
    print("✓ Step 2 complete.")

    #βήμα 3
    print("\nRunning STEP 3: Collecting candidates")
    candidates = collect_candidates(bins, inverted) #συλλογή υποψηφίων για κάθε ερώτημα
    with open(outdir / "candidates.txt", "w") as f: #αποθήκευση υποψηφίων
        for i, cand in enumerate(candidates): #για κάθε ερώτημα
            f.write(f"Query {i}: {','.join(map(str, cand))}\n") #εγγραφή υποψηφίων
    print("✓ Step 3 complete.")

    #βήμα 4
    print("\nRunning STEP 4: Exact Search")

    range_mode = (args.range.lower() == "true") #έλεγχος αν είναι ενεργό το range search

    knn_results, range_results, approx_times = exact_search(
        X,
        Q,
        candidates,
        args.R,
        args.N,
        range_mode
    ) #εκτέλεση ακριβούς αναζήτησης

    #αποθήκευση αποτελεσμάτων
    out_file = outdir / "results.txt"
    with open(out_file, "w") as f: #άνοιγμα αρχείου για εγγραφή
        for qi in range(len(Q)): #για κάθε ερώτημα
            f.write(f"Query {qi}:\n") #εγγραφή κεφαλίδας ερωτήματος

            #KNN (πάντα)
            f.write("KNN: ")
            f.write(",".join(f"{pid}:{dist:.4f}" for pid, dist in knn_results[qi]))
            f.write("\n")

            #RANGE (αν ενεργό)
            if range_mode:
                f.write("RANGE: ")
                f.write(",".join(f"{pid}:{dist:.4f}" for pid, dist in range_results[qi]))
                f.write("\n")
            
            f.write("\n")


    print(f"✓ Step 4 complete. Results saved to {out_file}")

    #βήμα 5
    print("\nRunning STEP 5: Writing final output")

    final_output_path = outdir / "final_output.txt"

    write_output_file(
        final_output_path,
        X,
        Q,
        knn_results,
        range_results,
        args.N,
        args.R,
        "Neural LSH",
        approx_times,
        range_mode
    ) #εγγραφή τελικού αρχείου εξόδου


    print(f"✓ Step 5 complete. Final output written to {final_output_path}")

    return {
        "probs": all_probs,
        "bins": bins,
        "candidates": candidates
    }
