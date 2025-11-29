import argparse #για την ανάλυση των ορισμάτων της γραμμής εντολών
from pipeline.builder import build_pipeline #για την κατασκευή pipeline ευρετηρίου

#κύρια συνάρτηση
def main():
    #ανάλυση ορισμάτων γραμμής εντολών
    parser = argparse.ArgumentParser(description="NLSH build pipeline")

    # --- βασικά flags της εκφώνησης ---
    parser.add_argument("-d", "--data", required=True,
                        help="Input dataset file (e.g., input.dat)")
    parser.add_argument("-i", "--index_path", required=True,
                        help="Output index path (e.g., nlsh_index)")
    parser.add_argument("-type", required=True, choices=["mnist", "sift"],
                        help="Dataset type")

    # --- παράμετροι kNN ---
    parser.add_argument("--knn", type=int, default=10)

    # --- KaHIP parameters ---
    parser.add_argument("-m", "--nblocks", type=int, default=100)
    parser.add_argument("--imbalance", type=float, default=0.03)
    parser.add_argument("--kahip_mode", type=int, default=2)

    # --- MLP parameters ---
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args() #ανάλυση ορισμάτων

    # ------------------------------------------------------
    #           Πλήρης κλήση pipeline
    # ------------------------------------------------------
    build_pipeline(
        dataset_type=args.type,
        dataset_path=args.data,
        index_path=args.index_path,
        k=args.knn,
        nblocks=args.nblocks,
        imbalance=args.imbalance,
        kahip_mode=args.kahip_mode,
        layers=args.layers,
        nodes=args.nodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    ) #κατασκευή pipeline ευρετηρίου


if __name__ == "__main__": #εκτέλεση κύριας συνάρτησης
    main()
