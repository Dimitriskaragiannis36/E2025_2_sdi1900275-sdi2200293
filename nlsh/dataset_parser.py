import numpy as np
from pathlib import Path

def load_mnist(path: str, max_images: int = -1, normalize: bool = True):
    """
    Φόρτωση MNIST .dat αρχείων.
    path: αρχείο input ή query
    max_images: αν >0, περιορίζει τον αριθμό εικόνων
    normalize: αν True, κλίμακα 0-1
    Επιστρέφει: np.ndarray (num_samples, 28*28), dtype=float32
    """
    data = np.fromfile(path, dtype=np.uint8)
    total_images = data.size // (28*28)
    count = total_images if max_images < 0 else min(max_images, total_images)
    data = data[:count*28*28].reshape(count, 28*28).astype(np.float32)
    if normalize:
        data /= 255.0
    return data

def load_sift(path: str, max_vectors: int = -1, expected_dim: int = 128):
    """
    Φόρτωση SIFT .dat/.fvecs αρχείων (Little-Endian)
    path: αρχείο input ή query
    max_vectors: αν >0, περιορίζει τον αριθμό διανυσμάτων
    expected_dim: συνήθως 128
    Επιστρέφει: np.ndarray (num_vectors, expected_dim), dtype=float32
    """
    vectors = []
    with open(path, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # EOF
            dim = int(np.frombuffer(dim_bytes, dtype=np.int32)[0])
            if dim != expected_dim:
                raise ValueError(f"Unexpected vector dimension: {dim} != {expected_dim}")
            vec_bytes = f.read(4*dim)
            if len(vec_bytes) != 4*dim:
                raise ValueError("Truncated SIFT vector in file")
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors.append(vec)
            if 0 < max_vectors == len(vectors):
                break
    if not vectors:
        print(f"Warning: no vectors read from {path}")
    return np.array(vectors, dtype=np.float32)

def load_dataset(dataset_type: str, split: str = "input", max_items: int = -1, normalize_mnist: bool = True):
    """
    Ενιαία συνάρτηση για MNIST ή SIFT
    dataset_type: "mnist" ή "sift"
    split: "input" ή "query"
    max_items: μέγιστος αριθμός δειγμάτων
    normalize_mnist: μόνο για MNIST
    """
    path = Path("data") / dataset_type / f"{split}.dat"
    if dataset_type.lower() == "mnist":
        return load_mnist(path, max_images=max_items, normalize=normalize_mnist)
    elif dataset_type.lower() == "sift":
        return load_sift(path, max_vectors=max_items)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
