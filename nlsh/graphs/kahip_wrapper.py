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
               nblocks, imbalance, kahip_mode, seed, suppress_output=True):

    kahip = _ensure_kahip()
    print("DEBUG: kahip_mode =", kahip_mode)

    xadj_i = [int(x) for x in xadj]
    adjncy_i = [int(x) for x in adjncy]
    adjcwgt_i = [int(x) for x in adjcwgt]
    vwgt_i = [int(x) for x in vwgt]

    edgecut, blocks = kahip.kaffpa(
        vwgt_i, xadj_i, adjcwgt_i, adjncy_i,
        int(nblocks), float(imbalance),
        bool(suppress_output), int(seed), int(kahip_mode)
    )

    return edgecut, blocks


def partition_knn_graph(graph, nblocks, imbalance, kahip_mode, seed, nodes):
    if nodes is None:
        nodes = gp._detect_n_nodes(graph)
    print("DEBUG: nodes =", nodes)
    print("DEBUG: seed =", seed)
    adj = gp.build_symmetric_weighted_adj_from_directed(graph, nodes)
    xadj, adjncy, adjcwgt, vwgt = cu.adj_to_csr(adj)
    edgecut, blocks = call_kahip(xadj, adjncy, adjcwgt, vwgt, nblocks, imbalance, kahip_mode, seed)
    print("DEBUG: m =", nblocks)
    print("DEBUG: imbalance =", imbalance)
    if len(blocks) < nodes:
        blocks = list(blocks) + [-1] * (nodes - len(blocks))
    else:
        blocks = list(blocks)[:nodes]

    parts_map = defaultdict(list)
    for node_id, part in enumerate(blocks):
        parts_map[int(part)].append(node_id)

    return blocks, dict(parts_map), {"edgecut": edgecut, "xadj": xadj, "adjncy": adjncy, "adjcwgt": adjcwgt, "vwgt": vwgt}
