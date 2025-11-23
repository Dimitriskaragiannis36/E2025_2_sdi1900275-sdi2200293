import torch
import json
import numpy as np
from ml.classifier import MLPClassifier

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def load_model(model_path, config):
    model = MLPClassifier(
        in_dim=config["input_dim"],
        out_dim=config["num_classes"]
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def load_query(path):
    q = np.fromfile(path, dtype=np.float32)
    return torch.tensor(q, dtype=torch.float32)
