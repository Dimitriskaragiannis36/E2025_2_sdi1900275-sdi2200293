# kahip_partition.py
"""
Χρήση:
    from kahip_partition import partition_knn_graph

    blocks, parts_map = partition_knn_graph(
        graph,            # dict: node -> list(neighbor_ids)
        nblocks=100,
        imbalance=0.03,
        mode=1,           # 0 FAST, 1 ECO, 2 STRONG
        seed=1,
        write_prefix="nlsh_kahip"  # optional prefix για αποθήκευση αρχείων
    )

Εξαγωγές:
    blocks: list[int] κατά node-id (blocks[node] = part)
    parts_map: dict part -> [node, ...] (inverted file)
    Επιπλέον δημιουργεί αρχεία:
      - {write_prefix}_partitions.txt  (node_id, part)
      - {write_prefix}_inverted.csv    (part, node0, node1, ...)
"""
import os
import sys
from collections import defaultdict

def _ensure_kahip():
    try:
        import kahip
        return kahip
    except Exception as e:
        raise ImportError(
            "Could not import kahip. Install via `pip install kahip` "
            "or ensure kahip is available. Original error: " + str(e)
        )

def _detect_n_nodes(graph):
    """Επιστρέφει n (αριθμό κόμβων). Αν το graph έχει μη διαδοχικά ids, υπολογίζουμε max+1."""
    if not graph:
        return 0
    max_id = max(graph.keys())
    return max_id + 1

def build_symmetric_weighted_adj_from_directed(graph, n_nodes=None):
    """
    graph: dict node -> list(neighbor_ids)   (directed k-NN)
    Επιστρέφει adjacency as dict: node -> dict(neighbor -> weight)
    Βάρη: 2 αν mutual, 1 αν one-sided.
    Διασφαλίζει συμμετρία (αν i->j τότε j->i υπάρχει με ίδιο βάρος).
    """
    if n_nodes is None:
        n_nodes = _detect_n_nodes(graph)

    # αρχικά: directed edges with weight 1
    adj = [dict() for _ in range(n_nodes)]
    for u, neighs in graph.items():
        if u < 0:
            continue
        for v in neighs:
            if v < 0:
                continue
            # αν πολλαπλές εισόδους, κρατάμε μόνο μία (βάρος αρχικά 1)
            adj[u].setdefault(v, 1)

    # τώρα μετατρέπουμε βάρη σε 2 αν mutual και εξασφαλίζουμε συμμετρία
    for u in range(n_nodes):
        for v in list(adj[u].keys()):
            if v < 0 or v >= n_nodes:
                continue
            if u in adj[v]:
                # mutual -> βάρος 2
                adj[u][v] = 2
                adj[v][u] = 2
            else:
                # one-sided: βεβαιώνουμε αντίστροφη εγγραφή με ίδιο βάρος (so CSR will contain both directions)
                # αλλά δεν αναγκαστικά set adj[v][u] (κάποιες μορφές ζητούν full undirected duplication)
                # για KaHIP πρέπει το graph να είναι undirected: επομένως εγγράφουμε και την αντίστροφη με βάρος 1
                adj[v].setdefault(u, 1)

    # Final pass: make sure diagonal and self-loops removed
    for u in range(n_nodes):
        if u in adj[u]:
            del adj[u][u]

    return adj

def adj_to_csr(adj):
    """
    adj: list of dicts, κάθε idx -> {neighbor: weight, ...}
    Επιστρέφει xadj, adjncy, adjcwgt, vwgt ως python lists (τύποι int).
    CSR convention:
      xadj length = n+1, xadj[0]=0, xadj[i+1]=xadj[i] + deg(i)
      adjncy, adjcwgt have same ordering.
    """
    n = len(adj)
    xadj = [0] * (n + 1)
    adjncy = []
    adjcwgt = []
    vwgt = [1] * n  # per exercise: node weights = 1

    pos = 0
    for i in range(n):
        neighbors = adj[i]
        # optional: sort neighbors for determinism (not required)
        items = sorted(neighbors.items(), key=lambda x: x[0])
        for v, w in items:
            adjncy.append(int(v))
            adjcwgt.append(int(w))
            pos += 1
        xadj[i+1] = pos

    return xadj, adjncy, adjcwgt, vwgt

def call_kahip(xadj, adjncy, adjcwgt, vwgt, nblocks, imbalance=0.03, mode=1, seed=1, suppress_output=True):
    """
    Καλεί kahip.kaffpa και επιστρέφει edgecut, blocks (list of labels)
    mode: 0 FAST, 1 ECO, 2 STRONG
    """
    kahip = _ensure_kahip()
    # kahip.kaffpa signature: (vwgt, xadj, adjcwgt, adjncy, nblocks, imbalance, suppress_output, seed, mode)
    # κάποιες εκδόσεις δέχονται positional args, κάποιες keyword. Θα χρησιμοποιήσουμε positional για ασφάλεια.
    # ensure lists are ints
    xadj_i = list(map(int, xadj))
    adjncy_i = list(map(int, adjncy))
    adjcwgt_i = list(map(int, adjcwgt))
    vwgt_i = list(map(int, vwgt))

    # call
    edgecut, blocks = kahip.kaffpa(vwgt_i, xadj_i, adjcwgt_i, adjncy_i,
                                   int(nblocks), float(imbalance),
                                   bool(suppress_output), int(seed), int(mode))
    return edgecut, blocks

def partition_knn_graph(graph, nblocks=100, imbalance=0.03, mode=1, seed=1, write_prefix=None, n_nodes=None):
    """
    Ουσιαστική συνάρτηση για το Βήμα 2.
    graph: dict node -> list(neighbor_ids)
    Επιστρέφει: blocks (list per node), parts_map (dict part -> list nodes)
    Αν write_prefix δοθεί, γράφει αρχεία:
        {write_prefix}_partitions.txt  (node, part)
        {write_prefix}_inverted.csv    (part, node0, node1, ...)
    """
    if n_nodes is None:
        n_nodes = _detect_n_nodes(graph)
    # build symmetric weighted adjacency
    adj = build_symmetric_weighted_adj_from_directed(graph, n_nodes=n_nodes)

    # build CSR arrays
    xadj, adjncy, adjcwgt, vwgt = adj_to_csr(adj)

    # call kahip
    edgecut, blocks = call_kahip(xadj, adjncy, adjcwgt, vwgt, nblocks, imbalance, mode, seed)

    # blocks is a sequence of ints length n_nodes (hopefully)
    if len(blocks) < n_nodes:
        # pad with -1 (shouldn't happen usually)
        blocks = list(blocks) + [-1] * (n_nodes - len(blocks))
    else:
        blocks = list(blocks)[:n_nodes]

    # build inverted map
    parts_map = defaultdict(list)
    for node_id, part in enumerate(blocks):
        parts_map[int(part)].append(node_id)

    # write outputs if requested
    if write_prefix:
        p1 = f"{write_prefix}_partitions.txt"
        with open(p1, "w") as f:
            for node_id, part in enumerate(blocks):
                f.write(f"{node_id},{part}\n")
        p2 = f"{write_prefix}_inverted.csv"
        with open(p2, "w") as f:
            for part in sorted(parts_map.keys()):
                nodes = parts_map[part]
                f.write(",".join([str(part)] + [str(n) for n in nodes]) + "\n")

    return blocks, dict(parts_map), {"edgecut": edgecut, "xadj": xadj, "adjncy": adjncy, "adjcwgt": adjcwgt, "vwgt": vwgt}

# --- small CLI for direct run (handy for debug) ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build KaHIP partition from directed kNN graph CSV (node,dup,node,...).")
    parser.add_argument("--knn_csv", required=True, help="CSV produced by your search executable (one row per node)")
    parser.add_argument("--nblocks", type=int, default=100)
    parser.add_argument("--imbalance", type=float, default=0.03)
    parser.add_argument("--mode", type=int, default=1, choices=[0,1,2])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_prefix", default="nlsh_kahip")
    args = parser.parse_args()

    # simple CSV loader compatible with your graph_utils.load_knn_graph_from_csv format
    import csv
    graph = {}
    max_node = -1
    with open(args.knn_csv, "r") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            try:
                node = int(row[0])
                # neighbors start at index 2 in the format you used earlier
                neighs = [int(x) for x in row[2:] if x != ""]
                graph[node] = neighs
                if node > max_node:
                    max_node = node
            except Exception:
                continue
    n_nodes = max_node + 1
    print(f"Loaded graph with {n_nodes} nodes.")

    blocks, parts_map, meta = partition_knn_graph(graph, nblocks=args.nblocks, imbalance=args.imbalance, mode=args.mode, seed=args.seed, write_prefix=args.out_prefix, n_nodes=n_nodes)
    print("KaHIP finished. edgecut:", meta["edgecut"])
    print(f"Partitions saved to {args.out_prefix}_partitions.txt and {args.out_prefix}_inverted.csv")
