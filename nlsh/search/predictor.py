import torch.nn.functional as F
import torch

def predict_bins(model, query_vector):
    """
    Step 1: M(q) = softmax(fθ(q))
    """
    if query_vector.dim() == 1:
        query_vector = query_vector.unsqueeze(0)

    with torch.no_grad():
        logits = model(query_vector)      
        probs = F.softmax(logits, dim=1)  

    return probs.squeeze(0)               
