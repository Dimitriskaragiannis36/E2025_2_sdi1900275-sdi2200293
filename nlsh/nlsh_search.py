import argparse

from search.loader import load_model, load_config, load_query
from search.predictor import predict_bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--model", type=str, default="data/model.pt")
    parser.add_argument("--config", type=str, default="data/config.json")
    args = parser.parse_args()

    print("Loading config...")
    cfg = load_config(args.config)

    print("Loading model...")
    model = load_model(args.model, cfg)

    print("Loading query...")
    q = load_query(args.query)

    print("Running Step 1: prediction...")
    probs = predict_bins(model, q)

    print("Prediction probabilities:")
    print(probs)

    print("\n✓ Search pipeline finished Step 1 only.")


if __name__ == "__main__":
    main()
