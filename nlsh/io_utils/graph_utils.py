import csv
from typing import Dict, List

def load_knn_graph_from_csv(path: str) -> Dict[int, List[int]]:
    """Load a kNN graph stored in CSV form: node_id, k, n1, n2, ..., nk"""
    graph = {}

    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 3:
                continue

            node = int(row[0])
            neighbors = list(map(int, row[2:]))
            graph[node] = neighbors

    return graph

