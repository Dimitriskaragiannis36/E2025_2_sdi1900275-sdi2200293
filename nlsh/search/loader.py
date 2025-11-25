import json
from pathlib import Path
import numpy as np
import torch
from ml.classifier import MLPClassifier
from io_utils.dataset_parser import load_dataset as dp_load_dataset


def load_index_files(index_dir):
    index_dir = Path(index_dir)
    meta = json.load(open(index_dir / "meta.json"))

    partitions = []
    with open(index_dir / "partitions.txt", "r") as f:
        for line in f:
            if line.strip():
                _, part = line.strip().split(",")
                partitions.append(int(part))

    inverted = {}
    with open(index_dir / "inverted.csv", "r") as f:
        for line in f:
            if line.strip():
                nums = list(map(int, line.strip().split(",")))
                part = nums[0]
                inverted[part] = nums[1:]

    return meta, partitions, inverted


def load_model(model_path, meta):
    from ml.classifier import MLPClassifier
    model = MLPClassifier(
        in_dim=meta["dim"],
        out_dim=meta["nblocks"],
        layers=meta["layers"],
        nodes=meta["nodes"]
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model

def load_dataset(path, dataset_type):
    return dp_load_dataset(path, dataset_type, max_items=-1)


def load_query(path, dataset_type=None):
    if dataset_type.lower() == "mnist":
        #διαβάζουμε όλα τα queries
        arr = np.fromfile(path, dtype=np.uint8)
        magic, num, rows, cols = arr[:16].view(">u4")
        dim = rows * cols
        data = arr[16:].reshape(num, dim).astype(np.float32) / 255.0
        return torch.tensor(data, dtype=torch.float32)
    elif dataset_type.lower() == "sift":
        #για SIFT χρησιμοποιούμε ήδη load_dataset
        return torch.tensor(load_dataset(path, "sift"), dtype=torch.float32)
