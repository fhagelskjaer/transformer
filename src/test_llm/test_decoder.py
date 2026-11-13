"""Evaluate a trained decoder on held-out JSONL conversations."""

import argparse
from pathlib import Path

from .decoder import evaluate, load_model, make_loader, perplexity, read_conversations, resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/decoder.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    loader = make_loader(read_conversations(args.data), model.max_seq_length, args.batch_size)
    loss, accuracy = evaluate(model, loader, device)
    print(f"Test responses: {len(loader.dataset)}; truncated: {loader.dataset.truncated_responses}; "
          f"loss: {loss:.4f}; perplexity: {perplexity(loss):.2f}; byte-token accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
