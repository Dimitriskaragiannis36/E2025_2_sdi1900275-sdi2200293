from collections import defaultdict #για να φτιάξουμε το parts_map
import graphs.graph_processing as gp #για να φτιάξουμε το adjacency matrix
import graphs.csr_utils as cu   #για να μετατρέψουμε το adjacency matrix σε CSR format

#βεβαιωνόμαστε ότι οι δεσμοί του KaHIP είναι διαθέσιμοι
def _ensure_kahip():
    try:
        import kahip #εισαγωγή των δεσμών του KaHIP
        return kahip #επιστροφή του module kahip
    except Exception as e: #σε περίπτωση αποτυχίας
        raise ImportError(
            f"Failed to import KaHIP Python bindings. "
            f"Install with `pip install kahip`. Underlying error: {e}"
        )

#κλήση του KaHIP με τις παραμέτρους που δίνονται
def call_kahip(xadj, adjncy, adjcwgt, vwgt,
               nblocks, imbalance, kahip_mode, seed, suppress_output=True):

    kahip = _ensure_kahip() #βεβαιωνόμαστε ότι οι δεσμοί του KaHIP είναι διαθέσιμοι
    print("DEBUG: kahip_mode =", kahip_mode)

    xadj_i = [int(x) for x in xadj] #μετατροπή των λιστών σε ακέραιους
    adjncy_i = [int(x) for x in adjncy] #μετατροπή των λιστών σε ακέραιους
    adjcwgt_i = [int(x) for x in adjcwgt] #μετατροπή των λιστών σε ακέραιους
    vwgt_i = [int(x) for x in vwgt] #μετατροπή των λιστών σε ακέραιους

    edgecut, blocks = kahip.kaffpa( 
        vwgt_i, xadj_i, adjcwgt_i, adjncy_i,
        int(nblocks), float(imbalance),
        bool(suppress_output), int(seed), int(kahip_mode)
    ) #κλήση της συνάρτησης kaffpa του KaHIP με τις παραμέτρους που δίνονται

    return edgecut, blocks #επιστροφή του edgecut και των blocks

#κύρια συνάρτηση για την κατάτμηση του γράφου kNN χρησιμοποιώντας το KaHIP
def partition_knn_graph(graph, nblocks, imbalance, kahip_mode, seed, nodes):
    if nodes is None:
        nodes = gp._detect_n_nodes(graph) #εντοπισμός του αριθμού των κόμβων αν δεν δοθεί
    print("DEBUG: nodes =", nodes)
    print("DEBUG: seed =", seed)
    adj = gp.build_symmetric_weighted_adj_from_directed(graph, nodes) #κατασκευή συμμετρικού σταθμισμένου adjacency matrix από τον κατευθυνόμενο γράφο
    xadj, adjncy, adjcwgt, vwgt = cu.adj_to_csr(adj) #μετατροπή του adjacency matrix σε CSR format
    edgecut, blocks = call_kahip(xadj, adjncy, adjcwgt, vwgt, nblocks, imbalance, kahip_mode, seed) #κλήση του KaHIP για την κατάτμηση
    print("DEBUG: m =", nblocks)
    print("DEBUG: imbalance =", imbalance)
    if len(blocks) < nodes:
        blocks = list(blocks) + [-1] * (nodes - len(blocks)) #προσθήκη -1 για τους κόμβους που δεν έχουν ανατεθεί σε κάποιο block
    else:
        blocks = list(blocks)[:nodes] #περιορισμός του μεγέθους της λίστας blocks στους κόμβους που υπάρχουν

    parts_map = defaultdict(list) #αρχικοποίηση του parts_map
    for node_id, part in enumerate(blocks): #για κάθε κόμβο και το αντίστοιχο block
        parts_map[int(part)].append(node_id) #προσθήκη του κόμβου στη λίστα του αντίστοιχου block

    return blocks, dict(parts_map), {"edgecut": edgecut, "xadj": xadj, "adjncy": adjncy, "adjcwgt": adjcwgt, "vwgt": vwgt}
