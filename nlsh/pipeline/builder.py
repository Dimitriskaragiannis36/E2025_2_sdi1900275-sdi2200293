from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

from graphs.knn_builder import build_and_load_knn_graph
from graphs.kahip_wrapper import partition_knn_graph
from io_utils.index_writer import write_partitions_file, write_inverted_csv, write_meta
from io_utils.dataset_parser import load_dataset
from ml.classifier import MLPClassifier, train
from io_utils.exec_finder import autodetect_exec


def build_pipeline(
    dataset_type,
    dataset_path=None,
    index_path=None,
    k=10,
    nblocks=8,
    imbalance=0.03,
    kahip_mode=2,
    layers=3,
    nodes=64,
    epochs=10,
    batch_size=128,
    lr=1e-3,
    seed=1,
):
    #αυτόμματος εντοπισμός του ANN εκτελέσιμου
    ann_exec = autodetect_exec()
    if ann_exec is None:
        raise RuntimeError("ERROR: Could not locate 'search' ANN executable.")

    print(f"✔ Using ANN executable from builder.py: {ann_exec}")

    print(f"=== [1] Building kNN graph ===")
    graph, csv_path = build_and_load_knn_graph(dataset_type, k, "ivfflat", ann_exec)

    print(f"=== [2] Running KaHIP ===")
    blocks, parts_map, meta = partition_knn_graph(
        graph,
        nblocks=nblocks,
        imbalance=imbalance,
        mode=kahip_mode,
        seed=seed,
        write_prefix=None
    )

    out_dir = Path(index_path) if index_path else Path(f"data/{dataset_type}/kahip")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== [3] Saving index files ===")
    write_partitions_file(blocks, out_dir / "partitions.txt")
    write_inverted_csv(parts_map, out_dir / "inverted.csv")
    write_meta(meta, out_dir / "meta.json")

    print("=== [4] Training classifier ===")
    X = load_dataset(dataset_type, split="input", max_items=len(blocks))

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(blocks, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLPClassifier(in_dim=X.shape[1], out_dim=nblocks)
    train(model, loader, epochs=epochs, lr=lr, device="cpu")

    torch.save(model.state_dict(), out_dir / "model.pth")

    return graph, blocks, parts_map
