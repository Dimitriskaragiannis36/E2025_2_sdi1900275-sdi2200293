import os #για λειτουργικό σύστημα
import shutil #για λειτουργίες αρχείων

#εντοπισμός του εκτελέσιμου αρχείου αναζήτησης
def autodetect_exec():
    """
    Βρίσκει το search executable αυτόματα.
    """
    candidates = [
        "./bin/search",
        "bin/search",
        "../bin/search",
    ] #πιθανοί κατάλογοι για το εκτελέσιμο αρχείο
    for c in candidates: #έλεγχος κάθε υποψήφιου διαδρομής
        if os.path.exists(c): #αν το αρχείο υπάρχει
            return os.path.abspath(c) #επιστροφή της απόλυτης διαδρομής

    home = os.path.expanduser("~") #καθορισμός του αρχικού καταλόγου του χρήστη
    for root, _, files in os.walk(home): #περιήγηση σε όλους τους υποκαταλόγους
        if "search" in files: #αν βρεθεί το αρχείο
            return os.path.join(root, "search") #επιστροφή της διαδρομής του αρχείου

    found = shutil.which("search") #αναζήτηση στο PATH του συστήματος
    if found: #αν βρεθεί
        return found

    return None
