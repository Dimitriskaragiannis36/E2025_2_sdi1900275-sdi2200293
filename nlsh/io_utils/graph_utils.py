import csv #για ανάγνωση αρχείων CSV
from typing import Dict, List #για τύπους δεδομένων

#φόρτωση ενός kNN γράφου από αρχείο CSV
def load_knn_graph_from_csv(path: str) -> Dict[int, List[int]]:
    """Load a kNN graph stored in CSV form: node_id, k, n1, n2, ..., nk"""
    graph = {} #αρχικοποίηση κενού γράφου

    with open(path, "r") as f: #άνοιγμα αρχείου για ανάγνωση
        reader = csv.reader(f) #δημιουργία αναγνώστη CSV
        for row in reader: #επανάληψη σε κάθε γραμμή του αρχείου
            if not row or row[0].startswith("#"): #παράλειψη κενών γραμμών ή σχολίων
                continue
            if len(row) < 3: #έλεγχος για επαρκή δεδομένα
                continue

            node = int(row[0]) #ανάγνωση του αναγνωριστικού κόμβου
            neighbors = list(map(int, row[2:])) #ανάγνωση των γειτόνων
            graph[node] = neighbors #προσθήκη στο γράφο

    return graph

