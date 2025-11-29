import torch.nn.functional as F #για softmax
import torch #για διαχείριση τανυστών

#πρόβλεψη των πιθανοτήτων των bins για ένα δοσμένο ερώτημα
def predict_bins(model, query_vector):
    """
    Step 1: M(q) = softmax(fθ(q))
    """
    if query_vector.dim() == 1: #προσθήκη batch dimension αν λείπει
        query_vector = query_vector.unsqueeze(0) #προσθήκη batch dimension

    with torch.no_grad(): #απενεργοποίηση υπολογισμού γραφημάτων
        logits = model(query_vector)     #λογίτες από το μοντέλο  
        probs = F.softmax(logits, dim=1)  #εφαρμογή softmax κατά μήκος της διάστασης των bins

    return probs.squeeze(0)          #αφαίρεση της διάστασης του batch πριν την επιστροφή      
