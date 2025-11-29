import torch.nn as nn #για το νευρωνικό δίκτυο
import torch.optim as optim #για τον optimizer

#απλός MLP classifier
class MLPClassifier(nn.Module):
    #αρχικοποίηση
    def __init__(self, in_dim, out_dim, layers, nodes):
        """
        in_dim  : διαστάσεις εισόδου
        out_dim : αριθμός blocks (labels)
        layers  : συνολικός αριθμός layers του MLP
        nodes   : νευρώνες ανά hidden layer
        """
        print("DEBUG: layers =", layers)
        super().__init__() #κληση constructor της nn.Module

        modules = [] #λίστα με τα layers
        input_dim = in_dim #αρχική διάσταση εισόδου

        #χτίζουμε (layers - 1) hidden layers
        #π.χ. layers=3 -> 2 hidden + 1 output
        for _ in range(layers - 1):
            modules.append(nn.Linear(input_dim, nodes)) #γραμμικό layer
            modules.append(nn.ReLU()) #λειτουργία ενεργοποίησης ReLU
            input_dim = nodes #ενημέρωση διαστάσεων για το επόμενο layer

        #τελικό output layer
        modules.append(nn.Linear(input_dim, out_dim))

        self.net = nn.Sequential(*modules) #δημιουργία του MLP
    #πρόοδος εμπρός
    def forward(self, x):
        return self.net(x)

#εκπαίδευση του μοντέλου
def train(model, dataloader, epochs, lr, device):
    model.to(device) #μεταφορά μοντέλου στη συσκευή (CPU/GPU)
    print("DEBUG: lr =", lr)
    opt = optim.Adam(model.parameters(), lr) #Adam optimizer
    loss_fn = nn.CrossEntropyLoss() #λειτουργία απώλειας
    print("DEBUG: epochs =", epochs)
    for epoch in range(epochs): #για κάθε εποχή
        model.train() #ρύθμιση σε λειτουργία εκπαίδευσης
        total_loss = 0.0 #αρχικοποίηση συνολικής απώλειας

        for X, y in dataloader: #για κάθε batch δεδομένων
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad() #μηδενισμός των gradients
            logits = model(X) #πρόοδος εμπρός
            loss = loss_fn(logits, y) #υπολογισμός απώλειας
            loss.backward() #υπολογισμός gradients
            opt.step() #ενημέρωση βαρών

            total_loss += loss.item() #συσσώρευση απώλειας

        print(f"Epoch {epoch}: loss={total_loss/len(dataloader):.4f}")
