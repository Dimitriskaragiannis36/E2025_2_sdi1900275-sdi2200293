from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

from graphs.knn_builder import build_and_load_knn_graph
from graphs.kahip_wrapper import partition_knn_graph
from io_utils.index_writer import write_partitions_file, write_inverted_csv, write_meta
from io_utils.dataset_parser import load_dataset
from ml.classifier import MLPClassifier, train

def build_pipeline(dataset_type, k, method, ann_exec, nblocks=8, imbalance=0.03):

    print(f"=== [1] Building kNN graph ===")
    graph, csv_path = build_and_load_knn_graph(dataset_type, k, method, ann_exec)

    print(f"=== [2] Running KaHIP ===")
    blocks, parts_map, meta = partition_knn_graph(
        graph,
        nblocks=nblocks,
        imbalance=imbalance,
        mode=1,
        seed=1,
        write_prefix=None
    )

    out_dir = Path(f"data/{dataset_type}/kahip")
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
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MLPClassifier(in_dim=X.shape[1], out_dim=nblocks)
    train(model, loader, epochs=10, lr=1e-3, device="cpu")

    torch.save(model.state_dict(), out_dir / "model.pth")

    return graph, blocks, parts_map
