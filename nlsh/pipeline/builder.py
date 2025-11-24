from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

from graphs.knn_builder import build_and_load_knn_graph
from graphs.kahip_wrapper import partition_knn_graph
from io_utils.index_writer import write_partitions_file, write_inverted_csv, write_meta
from io_utils.dataset_parser import load_dataset
from ml.classifier import MLPClassifier, train
from ml.data_utils import prepare_training_data
from io_utils.exec_finder import autodetect_exec


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
    if precomputed_knn_csv is None:
        ann_exec = autodetect_exec()
        if ann_exec is None:
            raise RuntimeError("ERROR: Could not locate ANN executable.")
        print(f"✔ Using ANN executable: {ann_exec}")
    else:
        ann_exec = None
        print(f"✔ Using PRECOMPUTED KNN csv: {precomputed_knn_csv}")
    graph, _ = build_and_load_knn_graph(
    dataset_type,
    k,
    "ivfflat",
    ann_exec,
    input_path=dataset_path,
    csv_path=precomputed_knn_csv )

    print(f"=== [2] Running KaHIP ===")
    blocks, parts_map, meta = partition_knn_graph(
        graph,
        nblocks,
        imbalance,
        kahip_mode,
        seed,
        nodes=None
    )

    out_dir = Path(index_path) if index_path else Path(f"data/{dataset_type}/kahip")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== [3] Saving index files ===")
    write_partitions_file(blocks, out_dir / "partitions.txt")
    write_inverted_csv(parts_map, out_dir / "inverted.csv")

    X = load_dataset(dataset_path, dataset_type, max_items=len(blocks))
    meta["dim"] = X.shape[1]
    meta["n"] = len(X)
    meta["nblocks"] = nblocks
    print("DEBUG: len(X) =", len(X))
    print("DEBUG: len(blocks) =", len(blocks))

    write_meta(meta, out_dir / "meta.json")

    print("=== [4] Training classifier ===")
    loader = prepare_training_data(X, blocks, batch_size)

    model = MLPClassifier(X.shape[1], nblocks, layers, nodes)

    train(model, loader, epochs, lr=lr, device="cpu")

    torch.save(model.state_dict(), out_dir / "model.pth")

    return graph, blocks, parts_map
