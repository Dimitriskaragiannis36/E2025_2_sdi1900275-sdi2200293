import numpy as np
from pathlib import Path

def load_mnist(path: str, max_images: int = -1, normalize: bool = True):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MNIST file not found: {path}")

    #διάβασε το δυαδικό αρχείο
    data = np.fromfile(path, dtype=np.uint8)

    #έλεγχος εγκυρότητας αρχείου
    if data.size < 16:
        raise ValueError("MNIST file too small.")

    magic, num, rows, cols = data[:16].view(">u4")

    if magic != 2051:
        raise ValueError(f"Invalid MNIST magic: {magic}")

    dim = rows * cols

    pixel_data = data[16:]

    if pixel_data.size != num * dim:
        raise ValueError(
            f"Corrupted MNIST: expected {num*dim} pixels, found {pixel_data.size}"
        )

    count = num if max_images < 0 else min(max_images, num)

    pixel_data = pixel_data[:count * dim]
    arr = pixel_data.reshape(count, dim).astype(np.float32)

    if normalize:
        arr /= 255.0

    return arr

def load_sift(path: str, max_vectors: int = -1, expected_dim: int = 128):
    vectors = []
    with open(path, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if len(dim_bytes) == 0:
                break
            if len(dim_bytes) < 4:
                raise ValueError("Corrupted SIFT file: unexpected EOF while reading dimension.")

            dim = int(np.frombuffer(dim_bytes, dtype=np.int32)[0])
            if dim != expected_dim:
                raise ValueError(f"Unexpected vector dimension: {dim} != {expected_dim}")

            vec_bytes = f.read(4 * dim)
            if len(vec_bytes) < 4 * dim:
                raise ValueError("Corrupted SIFT file: unexpected EOF while reading vector data.")

            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors.append(vec)

            if 0 < max_vectors == len(vectors):
                break

    if not vectors:
        print(f"Warning: no vectors read from {path}")

    return np.vstack(vectors).astype(np.float32)


def load_dataset(path: str, dataset_type: str,
                 max_items, normalize_mnist: bool = False):

    path = Path(path)
    dataset_type_l = dataset_type.lower()

    if dataset_type_l == "mnist":
        return load_mnist(path, max_images = max_items, normalize=normalize_mnist)

    elif dataset_type_l == "sift":
        return load_sift(path, max_vectors = max_items)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
