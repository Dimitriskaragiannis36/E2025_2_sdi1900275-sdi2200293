import subprocess #για την εκτέλεση εξωτερικών εντολών
from pathlib import Path #για τη διαχείριση διαδρομών αρχείων
from io_utils.graph_utils import load_knn_graph_from_csv #για τη φόρτωση kNN γραφήματος από CSV

#εκτέλεση ANN για την κατασκευή kNN γραφήματος
def run_ann_knn(dataset_type, k, method="ivfflat",
                ann_exec="./bin/search", input_path=None,
                knn_graph=True): #αν θέλουμε να παραχθεί και το knn γραφημα

    ann_exec = Path(ann_exec).resolve() #μετατροπή σε απόλυτη διαδρομή
    if not ann_exec.exists():
        raise FileNotFoundError(f"Executable not found: {ann_exec}")

    if dataset_type.lower() not in {"mnist", "sift"}:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if input_path is None:
        raise ValueError("input_path cannot be None — build_pipeline must pass the dataset file.")
    input_path = Path(input_path).resolve() #μετατροπή σε απόλυτη διαδρομή

    output_knn = Path(f"knn_graph_{dataset_type}.csv").resolve() #αποθήκευση του kNN γραφήματος σε CSV

    METHOD_PARAMS = {
        "lsh":       ["-k", "4", "-L", "5", "-w", "4.0"],
        "hypercube": ["-kproj", "14", "-w", "4", "-M", "10", "-probes", "2"],
        "ivfflat":   ["-kclusters", "50", "-nprobe", "5"],
        "ivfpq":     ["-kclusters", "50", "-nprobe", "5", "-M", "16", "-nbits", "8"]
    } #παράμετροι για κάθε μέθοδο

    if method not in METHOD_PARAMS: #έλεγχος έγκυρης μεθόδου
        raise ValueError(f"Unknown ANN method: {method}")

    method_flag = {
        "lsh": "-lsh",
        "hypercube": "-hypercube",
        "ivfflat": "-ivfflat",
        "ivfpq": "-ivfpq"
    }[method] #αντιστοίχιση μεθόδου σε σημαία εντολής

    cmd = [
        str(ann_exec),
        "-d", str(input_path),
        "-q", str(input_path),
        "-N", str(k),
        "-type", dataset_type,
        method_flag,
        "-range", "true",
        "-seed", "1"
    ] + METHOD_PARAMS[method] #προσθήκη παραμέτρων μεθόδου

    if knn_graph: #θέλουμε να παραχθεί kNN γράφημα
        cmd.append("-knngraph")

    print("Running:", " ".join(cmd))

    #έλεγχος της εκτέλεσης της εντολής
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("=== ANN EXEC STDOUT ===")
        print(result.stdout)
        print("=== ANN EXEC STDERR ===")
        print(result.stderr)
        raise RuntimeError("ANN executable failed.")

    if not output_knn.exists():
        raise FileNotFoundError(f"KNN graph CSV not produced: {output_knn}")

    return str(output_knn)

#κατασκευή και φόρτωση kNN γραφήματος
def build_and_load_knn_graph(
    dataset_type,
    k,
    method,
    ann_exec,
    input_path,
    csv_path
    ):

    """
    Αν csv_path != None → φορτώνουμε απευθείας το αρχείο
    Αν csv_path == None → τρέχει ANN για να παράγει νέο CSV
    """

    # ------------------------------------------
    # MODE B: Έχω ήδη CSV, απλά κάνε load
    # ------------------------------------------
    if csv_path is not None:
        print(f" Loading existing kNN CSV: {csv_path}")
        graph = load_knn_graph_from_csv(csv_path) #φόρτωση kNN γραφήματος από CSV

        if not graph:
            raise ValueError("KNN graph is empty — CSV may be corrupt.")
        return graph, csv_path

    # ------------------------------------------
    # MODE A: Κανονικό ANN mode
    # ------------------------------------------
    print("=== Running ANN KNN Builder ===")
    print("DEBUG: knn number =", k)

    csv_path = run_ann_knn(dataset_type, k, method, ann_exec, input_path) #τρέχει ANN για να παράγει CSV
    print(f"✔ ANN produced CSV: {csv_path}")

    graph = load_knn_graph_from_csv(csv_path) #φόρτωση kNN γραφήματος από CSV

    if not graph:
        raise ValueError("KNN graph is empty — ANN executable may have failed.")

    return graph, csv_path

