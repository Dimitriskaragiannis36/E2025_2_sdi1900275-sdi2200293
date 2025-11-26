import numpy as np
from search.predictor import predict_bins


def compute_probabilities(model, queries):
    """STEP 1: compute probability vector for κάθε query."""
    all_probs = [predict_bins(model, q).numpy() for q in queries]
    return np.vstack(all_probs)


def select_top_T_bins(all_probs, T):
    """STEP 2: Multi-probe bin selection (top-T bins ανά query)."""
    selected_bins = []
    selected_probs = []

    for probs in all_probs:
        #T μεγαλύτερες τιμές
        top_idx = np.argpartition(probs, -T)[-T:]

        #ταξινόμηση φθίνουσα
        top_idx_sorted = top_idx[np.argsort(probs[top_idx])[::-1]]

        selected_bins.append(top_idx_sorted)
        selected_probs.append(probs[top_idx_sorted])

    return np.vstack(selected_bins), np.vstack(selected_probs)


def collect_candidates(selected_bins, inverted_index):
    """STEP 3: συλλογή υποψηφίων από τους επιλεγμένους κάδους."""
    all_candidates = []

    for bins_for_query in selected_bins:
        candidate_ids = set()
        for b in bins_for_query:
            if b in inverted_index:
                candidate_ids.update(inverted_index[b])
        all_candidates.append(sorted(candidate_ids))

    return all_candidates


def exact_search(X, Q, candidate_lists, R, N, range_mode=True):
    #Μετατροπή σε numpy arrays αν χρειάζεται
    if hasattr(X, "numpy"):
        X = X.numpy()
    if hasattr(Q, "numpy"):
        Q = Q.numpy()

    results = []

    for qi, q in enumerate(Q):
        cand = candidate_lists[qi]
        if not cand:
            results.append([])   #όχι υποψήφιοι
            continue

        pts = X[cand]                            #σχήμα [num_candidates, dim]
        diff = pts - q                           #απόσταση από το query
        dists = np.sum(diff * diff, axis=1)      #ευκλείδεια απόσταση στο τετράγωνο

        if range_mode:
            #κρατάμε μόνο όσους έχουν dist <= R
            mask = dists <= R
            kept_ids = np.array(cand)[mask]
            kept_dists = dists[mask]

            #ταξινόμηση κατά απόσταση
            order = np.argsort(kept_dists)
            final_ids = kept_ids[order].tolist()
            final_dists = kept_dists[order].tolist()
        else:
            #top-N κοντινότεροι
            if len(dists) > N:
                top_idx = np.argpartition(dists, N)[:N]
            else:
                top_idx = np.arange(len(dists))

            #ταξινόμηση κατά απόσταση
            order = np.argsort(dists[top_idx])
            final = top_idx[order]

            final_ids = np.array(cand)[final].tolist()
            final_dists = dists[top_idx][order].tolist()

        results.append(list(zip(final_ids, final_dists)))

    return results
