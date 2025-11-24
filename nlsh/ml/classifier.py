import torch
import torch.nn as nn
import torch.optim as optim

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, layers, nodes):
        """
        in_dim  : διαστάσεις εισόδου
        out_dim : αριθμός blocks (labels)
        layers  : συνολικός αριθμός layers του MLP
        nodes   : νευρώνες ανά hidden layer
        """
        print("DEBUG: layers =", layers)
        super().__init__()

        modules = []
        input_dim = in_dim

        #χτίζουμε (layers - 1) hidden layers
        #π.χ. layers=3 -> 2 hidden + 1 output
        for _ in range(layers - 1):
            modules.append(nn.Linear(input_dim, nodes))
            modules.append(nn.ReLU())
            input_dim = nodes

        #τελικό output layer
        modules.append(nn.Linear(input_dim, out_dim))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def train(model, dataloader, epochs, lr, device):
    model.to(device)
    print("DEBUG: lr =", lr)
    opt = optim.Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()
    print("DEBUG: epochs =", epochs)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss/len(dataloader):.4f}")
