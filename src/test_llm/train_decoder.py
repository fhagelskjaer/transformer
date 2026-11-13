"""Train a small decoder-only chatbot on JSONL conversations."""

import argparse
import math
from pathlib import Path

import torch

from .decoder import (FORMAT_VERSION, IGNORE_INDEX, Decoder, batch_loss, evaluate,
                      make_loader, perplexity, read_conversations, resolve_device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/decoder.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0 or args.max_length < 2 or args.embed_size < 1
            or args.layers < 1 or args.heads < 1 or args.embed_size % args.heads
            or not 0 <= args.dropout < 1):
        parser.error("Invalid hyperparameters; sizes and learning rate must be positive, "
                     "embed-size must be divisible by heads, and dropout must be in [0, 1)")
    if args.train_data.resolve() == args.validation_data.resolve():
        parser.error("Training and validation data must be different files")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    training = read_conversations(args.train_data)
    validation = read_conversations(args.validation_data)
    train_loader = make_loader(training, args.max_length, args.batch_size, shuffle=True)
    val_loader = make_loader(validation, args.max_length, args.batch_size)
    config = dict(embed_size=args.embed_size, num_layers=args.layers, num_heads=args.heads,
                  max_seq_length=args.max_length, dropout_p=args.dropout, ff_hidden_mult=4)
    model = Decoder(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"Device: {device}; training responses: {len(train_loader.dataset)}; "
          f"validation responses: {len(val_loader.dataset)}")
    print(f"Responses truncated to fit: training {train_loader.dataset.truncated_responses}, "
          f"validation {val_loader.dataset.truncated_responses}")
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_tokens = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss_sum = batch_loss(model, inputs, labels)
            count = (labels != IGNORE_INDEX).sum().item()
            loss = loss_sum / count
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            train_loss += loss_sum.item()
            train_tokens += count
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train loss {train_loss / train_tokens:.4f}, "
              f"validation loss {val_loss:.4f}, perplexity {perplexity(val_loss):.2f}, "
              f"byte-token accuracy {val_accuracy:.2%}")
        if val_loss < best_loss:
            best_loss = val_loss
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"format": "byte-dialogue", "version": FORMAT_VERSION, "config": config,
                        "model_state": model.state_dict(), "epoch": epoch}, args.checkpoint)
            print(f"Saved best checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
