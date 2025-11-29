import argparse #για την ανάλυση των ορισμάτων της γραμμής εντολών
from pathlib import Path #για τη διαχείριση διαδρομών αρχείων και φακέλων

from search.loader import load_model, load_index_files, load_query, load_dataset #εισαγωγή συναρτήσεων φόρτωσης δεδομένων
from search.runner import run_nlsh_search #εισαγωγή της συνάρτησης εκτέλεσης αναζήτησης

#κύρια συνάρτηση του προγράμματος
def main():
    parser = argparse.ArgumentParser() #δημιουργία αναλυτή ορισμάτων γραμμής εντολών

    parser.add_argument("-d", "--data", required=True)
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("-i", "--index", required=True)
    parser.add_argument("-o", "--output", required=True)

    parser.add_argument("-type", required=True, choices=["mnist", "sift"])
    parser.add_argument("-N", type=int, default=1)
    parser.add_argument("-R", type=float, default=None)
    parser.add_argument("-T", type=int, default=5)
    parser.add_argument("-range", type=str, default="true")

    args = parser.parse_args() #ανάλυση των ορισμάτων της γραμμής εντολών
 
    if args.R is None: #ορισμός προεπιλεγμένης τιμής για το R ανάλογα με τον τύπο δεδομένων
        args.R = 2000.0 if args.type == "mnist" else 2800.0
    
    index_dir = Path(args.index) #διαδρομή προς τον φάκελο του ευρετηρίου

    print("=== Loading index files ===")
    meta, inverted = load_index_files(index_dir) #φόρτωση αρχείων ευρετηρίου

    print("=== Loading model ===")
    model = load_model(index_dir / "model.pth", meta) #φόρτωση του εκπαιδευμένου μοντέλου

    print("=== Loading dataset & query ===")
    X = load_dataset(args.data, args.type, max_items= 10000) #φόρτωση του συνόλου δεδομένων
    Q = load_query(args.query, args.type, max_items= 100) #φόρτωση των ερωτημάτων αναζήτησης

    # ----------------------------------------------------------------------
    #δημιουργία φακέλου εξόδου
    # ----------------------------------------------------------------------
    base_output_dir = Path(args.output) #βασική διαδρομή φακέλου εξόδου
    final_out_dir = base_output_dir / args.type #διαδρομή φακέλου εξόδου ανά τύπο δεδομένων
    final_out_dir.mkdir(parents=True, exist_ok=True) #δημιουργία φακέλου εξόδου αν δεν υπάρχει
    print(f"Output directory: {final_out_dir}")
    # ----------------------------------------------------------------------

    run_nlsh_search(model, X, Q, inverted, args, final_out_dir) #εκτέλεση της αναζήτησης NLSH και αποθήκευση των αποτελεσμάτων


if __name__ == "__main__": #εκτέλεση της κύριας συνάρτησης αν το αρχείο εκτελείται ως κύριο πρόγραμμα
    main()
