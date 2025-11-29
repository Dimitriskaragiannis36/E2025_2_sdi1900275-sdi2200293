import torch #για να χρησιμοποιήσουμε PyTorch
from torch.utils.data import TensorDataset, DataLoader #για να δημιουργήσουμε σύνολα δεδομένων και φορτωτές δεδομένων

#προετοιμασία δεδομένων εκπαίδευσης
def prepare_training_data(X, labels, batch_size):
    print("DEBUG: batch_size =", batch_size)
    X_t = torch.tensor(X, dtype=torch.float32) #μετατροπή των δεδομένων εισόδου σε τανυστές PyTorch
    y_t = torch.tensor(labels, dtype=torch.long) #μετατροπή των ετικετών σε τανυστές PyTorch
    dataset = TensorDataset(X_t, y_t) #δημιουργία συνόλου δεδομένων
    loader = DataLoader(dataset, batch_size, shuffle=True) #δημιουργία φορτωτή δεδομένων με τυχαία ανάμειξη
    return loader
