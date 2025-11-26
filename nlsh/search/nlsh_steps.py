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
    """
    STEP 3: From the selected T bins per query, collect all candidate point IDs
    using the inverted index (loaded from inverted.csv).
    
    selected_bins: shape (num_queries, T)
    inverted_index: dict[int, list[int]]
    """

    all_candidates = []

    for bins_for_query in selected_bins:
        candidate_ids = set()
        for b in bins_for_query:
            if b in inverted_index:
                candidate_ids.update(inverted_index[b])
        all_candidates.append(sorted(candidate_ids))

    return all_candidates

