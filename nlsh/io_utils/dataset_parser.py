import numpy as np
from pathlib import Path

def load_mnist(path: str, max_images: int = -1, normalize: bool = True):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MNIST file not found: {path}")

    data = np.fromfile(path, dtype=np.uint8)

    if data.size % (28*28) != 0:
        raise ValueError(f"Corrupted MNIST file: size {data.size} is not divisible by 28*28.")

    total_images = data.size // (28*28)
    count = total_images if max_images < 0 else min(max_images, total_images)

    data = data[:count*28*28].reshape(count, 28*28).astype(np.float32)

    if normalize:
        data /= 255.0

    return data


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


def load_dataset(dataset_type: str, split: str = "input",
                 max_items: int = -1, normalize_mnist: bool = True):

    dataset_type_l = dataset_type.lower()
    path = Path("data") / dataset_type_l / f"{split}.dat"

    if dataset_type_l == "mnist":
        return load_mnist(path, max_images=max_items, normalize=normalize_mnist)
    elif dataset_type_l == "sift":
        return load_sift(path, max_vectors=max_items)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
