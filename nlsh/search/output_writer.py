import time #χρειαζόμαστε για τη μέτρηση χρόνου
import numpy as np #χρειαζόμαστε για αριθμητικούς υπολογισμούς

import torch #χρειαζόμαστε για χειρισμό tensores

#υπολογισμός των πραγματικών γειτόνων με πλήρη αναζήτηση
def full_search_true_neighbors(X, q, N):
    #μετατροπή σε numpy arrays αν χρειάζεται
    if isinstance(q, torch.Tensor):
        q = q.numpy()

    if isinstance(X, torch.Tensor):
        X = X.numpy()

    diff = X - q #numpy array
    dists = np.sqrt(np.sum(diff * diff, axis=1)) #numpy array

    order = np.argsort(dists) #numpy array
    top_ids = order[:N].tolist() #λίστα με τα N πρώτα ids
    top_dists = dists[top_ids].tolist() #λίστα με τις αποστάσεις των N πρώτων

    #dists: **κρατάμε ως numpy array**
    return top_ids, top_dists


def write_output_file(
    out_path, 
    X, Q,
    knn_results,       #λίστα λιστών με (id, dist) από το STEP 4
    range_results,     #λίστα λιστών με (id, dist) ή []
    N, R,
    method_name,
    approx_times,
    range_mode
): #γράψε τα αποτελέσματα σε αρχείο κειμένου
    if approx_times is None:
        approx_times = [] #λίστα με χρόνους προσέγγισης

    num_queries = len(Q) #αριθμός ερωτημάτων
    AF_list = [] #λίστα με τους παράγοντες επιτάχυνσης
    recall_list = [] #λίστα με τις τιμές ανάκλησης
    true_times = [] #λίστα με χρόνους πλήρους αναζήτησης

    with open(out_path, "w") as f: #άνοιγμα αρχείου για εγγραφή
        f.write(f"{method_name}\n") #μέθοδος που χρησιμοποιήθηκε

        for qi, q in enumerate(Q): #για κάθε ερώτημα

            f.write(f"\nQuery: {qi}\n") #αριθμός ερωτήματος

            #προσεγγιστικοί γείτονες
            approx_neighbors = knn_results[qi][:N] #λίστα με (id, dist)
            approx_ids = [p[0] for p in approx_neighbors] #ids
            approx_dists = [p[1] for p in approx_neighbors] #αποστάσεις

            #πραγματικοί γείτονες
            t0 = time.perf_counter() #χρόνος έναρξης
            true_ids, true_dists = full_search_true_neighbors(X, q, N) #πραγματικοί γείτονες
            t1 = time.perf_counter() #χρόνος λήξης
            true_times.append(t1 - t0) #αποθήκευση χρόνου πλήρους αναζήτησης

            #δικλείδα ασφαλείας
            for i in range(N):
                if i < len(approx_ids): #υπάρχει προσεγγιστικός γείτονας
                    nn_id = approx_ids[i] #id
                    nn_dist = approx_dists[i] #απόσταση
                else:
                    nn_id = -1 #δεν υπάρχει προσεγγιστικός γείτονας
                    nn_dist = float("inf") #άπειρη απόσταση

                #πραγματική απόσταση
                true_dist = true_dists[i]

                f.write(f"Nearest neighbor-{i+1}: {nn_id}\n")
                f.write(f"distanceApproximate: {nn_dist}\n")
                f.write(f"distanceTrue: {true_dist}\n")

            #γείτονες εντός R
            if range_mode:
                f.write("R-near neighbors:\n")
                ann_range = range_results[qi] #λίστα με (id, dist) ή []
                for (rid, _) in ann_range: #για κάθε γείτονα εντός R
                    f.write(f"{rid}\n") #γράψε το id

            #μετρικές
            if approx_dists:
                AF = approx_dists[0] / true_dists[0] if true_dists[0] > 0 else float("inf") #παράγοντας επιτάχυνσης
            else:
                AF = float("inf")
            AF_list.append(AF) #αποθήκευση

            recall = len(set(approx_ids).intersection(true_ids)) / N #ανάκληση
            recall_list.append(recall) #αποθήκευση

        #μέσοι όροι
        avg_AF = float(np.mean(AF_list)) #μέσος όρος παράγοντα επιτάχυνσης
        avg_recall = float(np.mean(recall_list)) #μέσος όρος ανάκλησης

        tApproxAvg = float(np.mean(approx_times)) if approx_times else 0 #μέσος χρόνος προσέγγισης
        tTrueAvg = float(np.mean(true_times)) #μέσος χρόνος πλήρους αναζήτησης

        QPS = num_queries / np.sum(approx_times) if approx_times else 0 #ερωτήματα ανά δευτερόλεπτο

        f.write(f"\nAverage AF: {avg_AF}\n")
        f.write(f"Recall@N: {avg_recall}\n")
        f.write(f"QPS: {QPS}\n")
        f.write(f"tApproximateAverage: {tApproxAvg}\n")
        f.write(f"tTrueAverage: {tTrueAvg}\n")

