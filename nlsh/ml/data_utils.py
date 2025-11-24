import torch
from torch.utils.data import TensorDataset, DataLoader

def prepare_training_data(X, labels, batch_size):
    print("DEBUG: batch_size =", batch_size)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    return loader
