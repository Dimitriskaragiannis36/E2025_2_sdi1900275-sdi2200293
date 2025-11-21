"""
KaHIP wrapper that uses graph_processing and csr_utils.
"""
from collections import defaultdict
import graphs.graph_processing as gp
import graphs.csr_utils as cu  

def _ensure_kahip():
    try:
        import kahip
        return kahip
    except Exception as e:
        raise ImportError(
            f"Failed to import KaHIP Python bindings. "
            f"Install with `pip install kahip`. Underlying error: {e}"
        )

def call_kahip(xadj, adjncy, adjcwgt, vwgt,
               nblocks, imbalance=0.03, mode=1, seed=1, suppress_output=True):

    kahip = _ensure_kahip()

    xadj_i = [int(x) for x in xadj]
    adjncy_i = [int(x) for x in adjncy]
    adjcwgt_i = [int(x) for x in adjcwgt]
    vwgt_i = [int(x) for x in vwgt]

    edgecut, blocks = kahip.kaffpa(
        vwgt_i, xadj_i, adjcwgt_i, adjncy_i,
        int(nblocks), float(imbalance),
        bool(suppress_output), int(seed), int(mode)
    )

    return edgecut, blocks


def partition_knn_graph(graph, nblocks=100, imbalance=0.03, mode=1, seed=1, write_prefix=None, n_nodes=None):
    if n_nodes is None:
        n_nodes = gp._detect_n_nodes(graph)
    adj = gp.build_symmetric_weighted_adj_from_directed(graph, n_nodes=n_nodes)
    xadj, adjncy, adjcwgt, vwgt = cu.adj_to_csr(adj)
    edgecut, blocks = call_kahip(xadj, adjncy, adjcwgt, vwgt, nblocks, imbalance, mode, seed)
    if len(blocks) < n_nodes:
        blocks = list(blocks) + [-1] * (n_nodes - len(blocks))
    else:
        blocks = list(blocks)[:n_nodes]

    parts_map = defaultdict(list)
    for node_id, part in enumerate(blocks):
        parts_map[int(part)].append(node_id)

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
