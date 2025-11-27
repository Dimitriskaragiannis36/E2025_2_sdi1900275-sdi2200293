import time
import numpy as np

import torch

def full_search_true_neighbors(X, q, N):
    #μετατροπή σε numpy arrays αν χρειάζεται
    if isinstance(q, torch.Tensor):
        q = q.numpy()

    if isinstance(X, torch.Tensor):
        X = X.numpy()

    diff = X - q
    dists = np.sum(diff * diff, axis=1)   #numpy array

    order = np.argsort(dists)
    top_ids = order[:N].tolist()
    top_dists = dists[top_ids].tolist()

    #dists: **κρατάμε ως numpy array**
    return top_ids, top_dists, dists


def write_output_file(
    out_path, 
    X, Q,
    knn_results,       #λίστα λιστών με (id, dist) από το STEP 4
    range_results,     #λίστα λιστών με (id, dist) ή []
    N, R,
    method_name,
    approx_times,
    range_mode
):
    if approx_times is None:
        approx_times = []

    num_queries = len(Q)
    AF_list = []
    recall_list = []
    true_times = []

    with open(out_path, "w") as f:
        f.write(f"{method_name}\n")

        for qi, q in enumerate(Q):

            f.write(f"\nQuery: {qi}\n")

            #προσεγγιστικοί γείτονες
            approx_neighbors = knn_results[qi][:N]
            approx_ids = [p[0] for p in approx_neighbors]
            approx_dists = [p[1] for p in approx_neighbors]

            #πραγματικοί γείτονες
            t0 = time.time()
            true_ids, true_dists, all_true_dists = full_search_true_neighbors(X, q, N)
            t1 = time.time()
            true_times.append(t1 - t0)

            #δικλείδα ασφαλείας
            for i in range(N):
                if i < len(approx_ids):
                    nn_id = approx_ids[i]
                    nn_dist = approx_dists[i]
                else:
                    nn_id = -1
                    nn_dist = float("inf")

                #πραγματική απόσταση
                true_dist = true_dists[i]

                f.write(f"Nearest neighbor-{i+1}: {nn_id}\n")
                f.write(f"distanceApproximate: {nn_dist}\n")
                f.write(f"distanceTrue: {true_dist}\n")

            #γείτονες εντός R
            if range_mode:
                f.write("R-near neighbors:\n")
                ann_range = range_results[qi]
                for (rid, _) in ann_range:
                    f.write(f"{rid}\n")

            #μετρικές
            if approx_dists:
                AF = approx_dists[0] / true_dists[0] if true_dists[0] > 0 else float("inf")
            else:
                AF = float("inf")
            AF_list.append(AF)

            recall = len(set(approx_ids).intersection(true_ids)) / N
            recall_list.append(recall)

        #μέσοι όροι
        avg_AF = float(np.mean(AF_list))
        avg_recall = float(np.mean(recall_list))

        tApproxAvg = float(np.mean(approx_times)) if approx_times else 0
        tTrueAvg = float(np.mean(true_times))

        QPS = num_queries / np.sum(approx_times) if approx_times else 0

        f.write(f"\nAverage AF: {avg_AF}\n")
        f.write(f"Recall@N: {avg_recall}\n")
        f.write(f"QPS: {QPS}\n")
        f.write(f"tApproximateAverage: {tApproxAvg}\n")
        f.write(f"tTrueAverage: {tTrueAvg}\n")

