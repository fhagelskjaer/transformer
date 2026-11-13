"""Evaluate a saved IMDb classifier or predict a single review."""

import argparse
from pathlib import Path

import torch

from .sentiment import LABELS, evaluate, make_loader, read_reviews
from .transformers import Transformer, create_padding_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/imdb.pt"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path, help="Path to the extracted aclImdb directory")
    source.add_argument("--text", help="A single review to classify")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = Transformer(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    vocabulary = checkpoint["vocabulary"]
    max_length = checkpoint["config"]["max_seq_length"]
    if args.data_dir is not None:
        reviews = read_reviews(args.data_dir, "test")
        loader = make_loader(reviews, vocabulary, max_length, args.batch_size)
        loss, accuracy = evaluate(model, loader, device)
        print(f"Test reviews: {len(reviews)}; loss: {loss:.4f}; accuracy: {accuracy:.2%}")
    else:
        loader = make_loader([(args.text, 0)], vocabulary, max_length, 1)
        inputs, _ = next(iter(loader))
        inputs = inputs.to(device)
        with torch.no_grad():
            probabilities = model(inputs, mask=create_padding_mask(inputs)).softmax(dim=-1)[0]
        print(f"Prediction: {LABELS[probabilities.argmax().item()]} "
              f"(negative: {probabilities[0].item():.2%}, positive: {probabilities[1].item():.2%})")


if __name__ == "__main__":
    main()
