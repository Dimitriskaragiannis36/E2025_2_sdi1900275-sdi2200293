import subprocess
import os

def run_ann_knn(dataset_type, k, method="ivfflat",
                ann_exec="./bin/search",
                input_path=None,
                knn_graph=True):

    ann_exec = os.path.abspath(ann_exec)
    if not os.path.exists(ann_exec):
        raise FileNotFoundError(f"Executable not found: {ann_exec}")

    if input_path is None:
        input_path = f"data/{dataset_type}/input.dat"
    input_path = os.path.abspath(input_path)

    # Προσαρμογή ονόματος εξόδου CSV
    output_knn = os.path.abspath(f"knn_graph_{dataset_type}.csv")

    # ----- DEFAULT PARAMS from E1 -----
    METHOD_PARAMS = {
        "lsh":      ["-k", "4", "-L", "5", "-w", "4.0"],
        "hypercube": ["-kproj", "14", "-w", "4", "-M", "10", "-probes", "2"],
        "ivfflat": ["-kclusters", "50", "-nprobe", "5"],
        "ivfpq":   ["-kclusters", "50", "-nprobe", "5", "-M", "16", "-nbits", "8"]
    }

    if method not in METHOD_PARAMS:
        raise ValueError("Unknown method " + method)

    method_flag = {
        "lsh": "-lsh",
        "hypercube": "-hypercube",
        "ivfflat": "-ivfflat",
        "ivfpq": "-ivfpq"
    }[method]

    cmd = [
        ann_exec,
        "-d", input_path,
        "-q", input_path,   # self-query
        "-N", str(k),
        "-type", dataset_type,
        method_flag,
        "-range", "false",
        "-seed", "1"
    ] + METHOD_PARAMS[method]

    if knn_graph:
        cmd.append("-knngraph")   # ενεργοποίηση KNN graph mode

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    return output_knn

from graph_utils import load_knn_graph_from_csv

def build_and_load_knn_graph(dataset_type, k=10, method="ivfflat",
                             ann_exec="./bin/search", input_path=None):
    """
    Τρέχει το search από την Εργασία 1, παράγει το CSV,
    και το φορτώνει σε Python ως dict.
    """
    csv_path = run_ann_knn(dataset_type, k, method, ann_exec, input_path)
    graph = load_knn_graph_from_csv(csv_path)
    return graph, csv_path
