"""
Graph processing helpers: build symmetric weighted adjacency from directed kNN graph.
"""
from typing import Dict, List, Optional

def _detect_n_nodes(graph: Dict[int, List[int]]) -> int:
    if not graph:
        return 0
    max_key = max(graph.keys())
    max_val = max((max(v) for v in graph.values() if v), default=0)
    return max(max_key, max_val) + 1

def build_symmetric_weighted_adj_from_directed(graph, nodes):
    if nodes is None:
        nodes = _detect_n_nodes(graph)

    adj = [dict() for _ in range(nodes)]

   #δημιουργία αρχικού προσανατολισμένου πίνακα γειτνίασης
    for u, neighs in graph.items():
        if u < 0 or u >= nodes:
            continue
        for v in neighs:
            if 0 <= v < nodes:
                adj[u][v] = 1

    #μετατροπή σε συμμετρικό πίνακα με βάρη
    for u in range(nodes):
        for v in list(adj[u]):
            if u in adj[v]:
                adj[u][v] = adj[v][u] = 2
            else:
                adj[v].setdefault(u, 1)

    #αφαίρεση αυτοσυνδέσεων
    for u in range(nodes):
        adj[u].pop(u, None)

    return adj

