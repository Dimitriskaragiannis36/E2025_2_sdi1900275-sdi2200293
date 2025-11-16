import csv

def load_knn_graph_from_csv(path):
    graph = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            node = int(row[0])
            neighbors = list(map(int, row[2:]))  # skip duplicated id
            graph[node] = neighbors
    return graph
