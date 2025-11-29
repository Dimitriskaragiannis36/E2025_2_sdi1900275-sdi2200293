import json #για φόρτωση αρχείων JSON
from pathlib import Path #για διαχείριση διαδρομών αρχείων
import numpy as np #για αριθμητικούς υπολογισμούς
import torch #για διαχείριση τανυστών και μοντέλων
from ml.classifier import MLPClassifier #για ορισμό MLP ταξινομητή
from io_utils.dataset_parser import load_dataset as dp_load_dataset #για φόρτωση dataset

#φόρτωση αρχείων ευρετηρίου
def load_index_files(index_dir):
    print("DEBUG:  index_path=", index_dir)
    index_dir = Path(index_dir) #μετατροπή σε αντικείμενο Path

    #meta.json
    meta = json.load(open(index_dir / "meta.json"))

    #inverted.csv
    inverted = {}
    with open(index_dir / "inverted.csv", "r") as f: #φόρτωση αντίστροφου ευρετηρίου
        for line in f:
            line = line.strip() #αφαίρεση κενών
            if not line:
                continue
            nums = list(map(int, line.split(","))) #μετατροπή σε λίστα ακεραίων
            part = nums[0] #πρώτος αριθμός είναι το part ID
            inverted[part] = nums[1:] #υπόλοιποι αριθμοί είναι τα IDs των αντικειμένων

    return meta, inverted

#φόρτωση εκπαιδευμένου μοντέλου MLP
def load_model(model_path, meta):
    model = MLPClassifier(
        in_dim=meta["dim"],
        out_dim=meta["nblocks"],
        layers=meta["layers"],
        nodes=meta["nodes"]
    ) #ορισμός μοντέλου MLP με παραμέτρους από meta.json
    state = torch.load(model_path, map_location="cpu", weights_only=True) #φόρτωση αποθηκευμένων βαρών
    model.load_state_dict(state) #φόρτωση βαρών στο μοντέλο
    model.eval() #ρύθμιση μοντέλου σε λειτουργία αξιολόγησης
    return model

#φόρτωση dataset
def load_dataset(path, dataset_type, max_items): #φόρτωση dataset με χρήση βοηθητικής συνάρτησης
    return dp_load_dataset(path, dataset_type, max_items)

#φόρτωση ερωτημάτων
def load_query(path, dataset_type, max_items):
    if dataset_type.lower() == "mnist": #ειδική περίπτωση για MNIST
        arr = np.fromfile(path, dtype=np.uint8) #ανάγνωση δυαδικού αρχείου
        magic, num, rows, cols = arr[:16].view(">u4") #ανάγνωση κεφαλίδας
        dim = rows * cols #διάσταση δεδομένων
        data = arr[16:].reshape(num, dim).astype(np.float32) / 255.0 #κανονικοποίηση δεδομένων
        if max_items > 0: #περιορισμός στον αριθμό αντικειμένων
            data = data[:max_items] #περιορισμός
        return torch.tensor(data, dtype=torch.float32)

    elif dataset_type.lower() == "sift": #ειδική περίπτωση για SIFT
        data = load_dataset(path, "sift", max_items) #φόρτωση με τη γενική συνάρτηση
        if max_items > 0: #περιορισμός στον αριθμό αντικειμένων
            data = data[:max_items] #περιορισμός
        return torch.tensor(data, dtype=torch.float32)

