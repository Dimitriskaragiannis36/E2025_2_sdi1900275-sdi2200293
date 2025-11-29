from typing import List, Dict, Tuple #απαραίτητο για τύπους

#μετατροπή λίστας γειτνίασης σε μορφή CSR
def adj_to_csr(adj: List[Dict[int,int]]) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Convert adjacency list-of-dicts to CSR arrays:
      xadj (len n+1), adjncy, adjcwgt, vwgt
    """
    n = len(adj) #αριθμός κόμβων
    xadj = [0] * (n + 1) #αρχικοποίηση xadj
    adjncy = [] #λίστα γειτονικών κόμβων
    adjcwgt = [] #λίστα βαρών ακμών
    vwgt = [1] * n  #βάρη κόμβων (προκαθορισμένα σε 1)

    pos = 0
    for i in range(n): #για κάθε κόμβο
        neighbors = adj[i] #λεξικό γειτόνων και βαρών
        items = sorted(neighbors.items(), key=lambda x: x[0]) #ταξινόμηση κατά κόμβο
        for v, w in items:
            adjncy.append(int(v)) #προσθήκη γείτονα
            adjcwgt.append(int(w)) #προσθήκη βάρους ακμής
            pos += 1 #ενημέρωση θέσης
        xadj[i+1] = pos #ενημέρωση xadj

    return xadj, adjncy, adjcwgt, vwgt #επιστροφή CSR δομών
