from typing import List, Dict, Tuple

def adj_to_csr(adj: List[Dict[int,int]]) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Convert adjacency list-of-dicts to CSR arrays:
      xadj (len n+1), adjncy, adjcwgt, vwgt
    """
    n = len(adj)
    xadj = [0] * (n + 1)
    adjncy = []
    adjcwgt = []
    vwgt = [1] * n  

    pos = 0
    for i in range(n):
        neighbors = adj[i]
        items = sorted(neighbors.items(), key=lambda x: x[0])
        for v, w in items:
            adjncy.append(int(v))
            adjcwgt.append(int(w))
            pos += 1
        xadj[i+1] = pos

    return xadj, adjncy, adjcwgt, vwgt
