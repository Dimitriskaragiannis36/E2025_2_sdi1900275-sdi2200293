from pathlib import Path #για μονοπάτια αρχείων
import torch #για αποθήκευση και εκπαίδευση μοντέλου

from graphs.knn_builder import build_and_load_knn_graph #για κατασκευή και φόρτωση knn γράφου
from graphs.kahip_wrapper import partition_knn_graph #για διαμέριση knn γράφου με KaHIP
from io_utils.index_writer import write_partitions_file, write_inverted_csv, write_meta #για αποθήκευση αρχείων ευρετηρίου
from io_utils.dataset_parser import load_dataset #για φόρτωση dataset
from ml.classifier import MLPClassifier, train #για ορισμό και εκπαίδευση MLP ταξινομητή
from ml.data_utils import prepare_training_data #για προετοιμασία δεδομένων εκπαίδευσης
from io_utils.exec_finder import autodetect_exec #για αυτόματη ανίχνευση εκτελέσιμου αρχείου ANN

#κατασκευή pipeline ευρετηρίου
def build_pipeline(
    dataset_type,
    dataset_path,
    index_path,
    k,
    nblocks,
    imbalance,
    kahip_mode,
    layers,
    nodes,
    epochs,
    batch_size,
    lr,
    seed,
    precomputed_knn_csv= None        #για COLAB "knn_graph_sift.csv" ή "knn_graph_mnist.csv"
):
    if precomputed_knn_csv is None: #χρήση ANN για κατασκευή knn γράφου
        ann_exec = autodetect_exec() #αυτόματη ανίχνευση εκτελέσιμου αρχείου ANN
        if ann_exec is None: #ανίχνευση απέτυχε
            raise RuntimeError("ERROR: Could not locate ANN executable.")
        print(f"✔ Using ANN executable: {ann_exec}")
    else:
        ann_exec = None #δεν χρειάζεται ANN εκτελέσιμο
        print(f"✔ Using PRECOMPUTED KNN csv: {precomputed_knn_csv}")
    graph, _ = build_and_load_knn_graph(
    dataset_type,
    k,
    "ivfflat",
    ann_exec,
    input_path=dataset_path,
    csv_path=precomputed_knn_csv ) #κατασκευή και φόρτωση knn γράφου

    print(f"=== [2] Running KaHIP ===")
    blocks, parts_map, meta = partition_knn_graph(
        graph,
        nblocks,
        imbalance,
        kahip_mode,
        seed,
        nodes=None
    ) #διαμέριση knn γράφου με KaHIP

    out_dir = Path(index_path) if index_path else Path(f"data/{dataset_type}/kahip") #καθορισμός φακέλου εξόδου
    out_dir.mkdir(parents=True, exist_ok=True) #δημιουργία φακέλου εξόδου αν δεν υπάρχει

    print("=== [3] Saving index files ===")
    write_partitions_file(blocks, out_dir / "partitions.txt") #αποθήκευση αρχείου διαμερίσεων
    write_inverted_csv(parts_map, out_dir / "inverted.csv") #αποθήκευση αντίστροφου ευρετηρίου

    X = load_dataset(dataset_path, dataset_type, max_items=len(blocks)) #φόρτωση dataset για εκπαίδευση ταξινομητή
    meta["dim"] = X.shape[1]
    meta["n"] = len(X)
    meta["nblocks"] = nblocks
    meta["layers"] = layers
    meta["nodes"] = nodes
    print("DEBUG: len(X) =", len(X))
    print("DEBUG: len(blocks) =", len(blocks))

    write_meta(meta, out_dir / "meta.json") #αποθήκευση αρχείου μεταδεδομένων

    print("=== [4] Training classifier ===")
    loader = prepare_training_data(X, blocks, batch_size) #προετοιμασία δεδομένων εκπαίδευσης

    model = MLPClassifier(X.shape[1], nblocks, layers, nodes) #ορισμός MLP ταξινομητή

    device = "cuda" if torch.cuda.is_available() else "cpu" #χρήση GPU αν είναι διαθέσιμη
    train(model, loader, epochs, lr=lr, device=device) #εκπαίδευση MLP ταξινομητή
    model.to("cpu")   #για αποθήκευση

    torch.save(model.state_dict(), out_dir / "model.pth") #αποθήκευση μοντέλου ταξινομητή

    return graph, blocks, parts_map
