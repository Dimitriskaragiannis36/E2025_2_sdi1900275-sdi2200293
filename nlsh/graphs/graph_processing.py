from typing import Dict, List #τύπος για γράφο

#ανίχνευση αριθμού κόμβων
def _detect_n_nodes(graph: Dict[int, List[int]]) -> int:
    if not graph: #κενός γράφος
        return 0
    max_key = max(graph.keys()) #μέγιστος κόμβος-κλειδί
    max_val = max((max(v) for v in graph.values() if v), default=0) #μέγιστος κόμβος-τιμή
    return max(max_key, max_val) + 1 #αριθμός κόμβων

#δημιουργία συμμετρικού πίνακα γειτνίασης με βάρη από προσανατολισμένο γράφο
def build_symmetric_weighted_adj_from_directed(graph, nodes):
    if nodes is None: #αν δεν δοθεί αριθμός κόμβων
        nodes = _detect_n_nodes(graph) #ανίχνευση αριθμού κόμβων

    adj = [dict() for _ in range(nodes)] #αρχικοποίηση λίστας γειτνίασης
 
   #δημιουργία αρχικού προσανατολισμένου πίνακα γειτνίασης
    for u, neighs in graph.items():
        if u < 0 or u >= nodes: #έλεγχος εγκυρότητας κόμβου
            continue
        for v in neighs: #για κάθε γείτονα
            if 0 <= v < nodes: #έλεγχος εγκυρότητας γείτονα
                adj[u][v] = 1 #προσθήκη ακμής με βάρος 1

    #μετατροπή σε συμμετρικό πίνακα με βάρη
    for u in range(nodes):
        for v in list(adj[u]): #για κάθε γείτονα του u
            if u in adj[v]: #αν υπάρχει αντίστροφη ακμή
                adj[u][v] = adj[v][u] = 2 #βάρος 2 για αμφίδρομες ακμές
            else:
                adj[v].setdefault(u, 1) #προσθήκη αντίστροφης ακμής με βάρος 1

    #αφαίρεση αυτοσυνδέσεων
    for u in range(nodes):
        adj[u].pop(u, None) #αφαίρεση αυτοσύνδεσης αν υπάρχει

    return adj

