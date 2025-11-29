import numpy as np #για αριθμητικούς υπολογισμούς
import time #για μέτρηση χρόνου
from search.predictor import predict_bins #για πρόβλεψη πιθανοτήτων κάδων

#υπολογισμός πιθανοτήτων για κάθε query
def compute_probabilities(model, queries):
    """STEP 1: compute probability vector for κάθε query."""
    all_probs = [predict_bins(model, q).numpy() for q in queries] #λίστα πιθανοτήτων
    return np.vstack(all_probs)

#επιλογή top-T κάδων βάσει πιθανοτήτων
def select_top_T_bins(all_probs, T):
    """STEP 2: Multi-probe bin selection (top-T bins ανά query)."""
    selected_bins = [] #λίστα επιλεγμένων κάδων
    selected_probs = [] #λίστα πιθανοτήτων για τους επιλεγμένους κάδους
    print("DEBUG:  T=", T)
    for probs in all_probs: #για κάθε διάνυσμα πιθανοτήτων
        #T μεγαλύτερες τιμές
        top_idx = np.argpartition(probs, -T)[-T:]

        #ταξινόμηση φθίνουσα
        top_idx_sorted = top_idx[np.argsort(probs[top_idx])[::-1]]

        selected_bins.append(top_idx_sorted) #προσθήκη επιλεγμένων κάδων
        selected_probs.append(probs[top_idx_sorted]) #προσθήκη πιθανοτήτων

    return np.vstack(selected_bins), np.vstack(selected_probs) #επιστροφή ως πίνακες numpy

#συλλογή υποψηφίων από τους επιλεγμένους κάδους
def collect_candidates(selected_bins, inverted_index):
    """STEP 3: συλλογή υποψηφίων από τους επιλεγμένους κάδους."""
    all_candidates = [] #λίστα υποψηφίων για κάθε query

    for bins_for_query in selected_bins: #για κάθε query
        candidate_ids = set() #σύνολο υποψηφίων (για αποφυγή διπλοεγγραφών)
        for b in bins_for_query: #για κάθε επιλεγμένο κάδο
            if b in inverted_index: #αν ο κάδος υπάρχει στον αντίστροφο δείκτη
                candidate_ids.update(inverted_index[b]) #προσθήκη υποψηφίων από τον κάδο
        all_candidates.append(sorted(candidate_ids)) #ταξινόμηση και προσθήκη στη λίστα

    return all_candidates

#ακριβής αναζήτηση μεταξύ υποψηφίων
def exact_search(X, Q, candidate_lists, R, N, range_mode):
    #Μετατροπή σε numpy arrays αν χρειάζεται
    if hasattr(X, "numpy"):
        X = X.numpy()
    if hasattr(Q, "numpy"):
        Q = Q.numpy()

    knn_results = []      #ΠΑΝΤΑ επιστρέφουμε N-NN
    range_results = []    #προαιρετικό (με βάση τη σημαία)
    approx_times = []     #χρόνος προσεγγιστικής αναζήτησης

    print("DEBUG: R =", R)
    print("DEBUG: N =", N)
    print("DEBUG: range =", range_mode)

    for qi, q in enumerate(Q): #για κάθε query
        cand = candidate_lists[qi] #λίστα υποψηφίων
        if not cand:
            knn_results.append([]) #κανένας υποψήφιος
            range_results.append([]) #κανένας υποψήφιος
            approx_times.append(0.0) #μηδενικός χρόνος
            continue
        t0 = time.perf_counter() #μέτρηση χρόνου αναζήτησης
        pts = X[cand]                            #σχήμα [num_candidates, dim]
        diff = pts - q                           #απόσταση από το query
        dists = np.sqrt(np.sum(diff * diff, axis=1))      #ευκλείδεια απόσταση 

        t1 = time.perf_counter() #τέλος μέτρησης χρόνου
        approx_times.append(t1 - t0) #αποθηκεύουμε τον χρόνο

        if len(dists) > N:
            top_idx = np.argpartition(dists, N)[:N] #N μικρότερες αποστάσεις
        else:
            top_idx = np.arange(len(dists)) #όλοι οι υποψήφιοι

        order_knn = np.argsort(dists[top_idx]) #ταξινόμηση k-NN
        final_knn = top_idx[order_knn] #τελικοί δείκτες k-NN

        knn_ids = np.array(cand)[final_knn].tolist() #ταυτότητες k-NN
        knn_dists = dists[top_idx][order_knn].tolist() #αποστάσεις k-NN

        knn_results.append(list(zip(knn_ids, knn_dists))) #αποθήκευση k-NN αποτελεσμάτων

        if range_mode:
            #κρατάμε μόνο όσους έχουν dist <= R
            mask = dists <= R #μάσκα για απόσταση εντός R
            kept_ids = np.array(cand)[mask] #ταυτότητες εντός R
            kept_dists = dists[mask] #αποστάσεις εντός R

            #ταξινόμηση κατά απόσταση
            order_range = np.argsort(kept_dists) #ταξινόμηση εντός R
            r_ids = kept_ids[order_range].tolist() #ταυτότητες εντός R ταξινομημένες
            r_dists = kept_dists[order_range].tolist()  #αποστάσεις εντός R ταξινομημένες

            range_results.append(list(zip(r_ids, r_dists))) #αποθήκευση αποτελεσμάτων εντός R
        else:
            range_results.append([]) #κενό αν δεν ζητείται

    return knn_results, range_results, approx_times
