import numpy as np #για αριθμητικούς υπολογισμούς
from pathlib import Path #για διαχείριση διαδρομών αρχείων

#φόρτωση και κανονικοποίηση του συνόλου δεδομένων MNIST
def load_mnist(path: str, max_images: int = -1, normalize: bool = True):
    path = Path(path) #μετατροπή σε αντικείμενο Path
    if not path.exists():
        raise FileNotFoundError(f"MNIST file not found: {path}")

    #διάβασε το δυαδικό αρχείο
    data = np.fromfile(path, dtype=np.uint8)

    #έλεγχος εγκυρότητας αρχείου
    if data.size < 16:
        raise ValueError("MNIST file too small.")
    #ανάγνωση κεφαλίδας
    magic, num, rows, cols = data[:16].view(">u4")

    if magic != 2051: #έλεγχος μαγικού αριθμού για εικόνες MNIST
        raise ValueError(f"Invalid MNIST magic: {magic}")

    dim = rows * cols #διάσταση εικόνας

    pixel_data = data[16:] #ανάγνωση δεδομένων pixel

    if pixel_data.size != num * dim: #έλεγχος πλήθους pixel
        raise ValueError(
            f"Corrupted MNIST: expected {num*dim} pixels, found {pixel_data.size}"
        )

    count = num if max_images < 0 else min(max_images, num) #προσδιορισμός πλήθους εικόνων προς φόρτωση

    pixel_data = pixel_data[:count * dim] #περιορισμός στα ζητούμενα δεδομένα
    arr = pixel_data.reshape(count, dim).astype(np.float32) #μετατροπή σε πίνακα NumPy

    if normalize: #κανονικοποίηση τιμών pixel
        arr /= 255.0 #τιμές στο διάστημα [0, 1]

    return arr

#φόρτωση συνόλου δεδομένων SIFT
def load_sift(path: str, max_vectors: int = -1, expected_dim: int = 128):
    vectors = [] #λίστα για αποθήκευση διανυσμάτων
    with open(path, "rb") as f: #άνοιγμα δυαδικού αρχείου
        while True:
            dim_bytes = f.read(4) #ανάγνωση διάστασης διανύσματος
            if len(dim_bytes) == 0:
                break
            if len(dim_bytes) < 4: #έλεγχος για απροσδόκητο EOF
                raise ValueError("Corrupted SIFT file: unexpected EOF while reading dimension.")

            dim = int(np.frombuffer(dim_bytes, dtype=np.int32)[0]) #μετατροπή σε ακέραιο
            if dim != expected_dim: #έλεγχος διάστασης
                raise ValueError(f"Unexpected vector dimension: {dim} != {expected_dim}")

            vec_bytes = f.read(4 * dim) #ανάγνωση δεδομένων διανύσματος
            if len(vec_bytes) < 4 * dim: #έλεγχος για απροσδόκητο EOF
                raise ValueError("Corrupted SIFT file: unexpected EOF while reading vector data.")

            vec = np.frombuffer(vec_bytes, dtype=np.float32) #μετατροπή σε πίνακα NumPy
            vectors.append(vec) #προσθήκη διανύσματος στη λίστα

            if 0 < max_vectors == len(vectors): #έλεγχος αν έχει φτάσει το μέγιστο πλήθος διανυσμάτων
                break

    if not vectors: #έλεγχος αν διαβάστηκαν διανύσματα
        print(f"Warning: no vectors read from {path}")

    return np.vstack(vectors).astype(np.float32) #επιστροφή ως πίνακας NumPy

#γενική συνάρτηση φόρτωσης συνόλου δεδομένων
def load_dataset(path: str, dataset_type: str,
                 max_items, normalize_mnist: bool = False):

    path = Path(path) #μετατροπή σε αντικείμενο Path
    dataset_type_l = dataset_type.lower() #μετατροπή τύπου συνόλου δεδομένων σε πεζά

    if dataset_type_l == "mnist": #φόρτωση συνόλου δεδομένων MNIST
        return load_mnist(path, max_images = max_items, normalize=normalize_mnist)

    elif dataset_type_l == "sift": #φόρτωση συνόλου δεδομένων SIFT
        return load_sift(path, max_vectors = max_items)

    else: #άγνωστος τύπος συνόλου δεδομένων
        raise ValueError(f"Unknown dataset type: {dataset_type}")
