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
