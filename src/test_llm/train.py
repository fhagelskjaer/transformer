"""Train the encoder classifier on the IMDb review dataset."""

import argparse
from pathlib import Path

import torch

from .sentiment import build_vocabulary, evaluate, make_loader, read_reviews, split_training_reviews
from .transformers import Transformer, create_padding_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the extracted aclImdb directory")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/imdb.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another PyTorch device")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0
            or args.max_length < 2 or args.vocab_size < 3 or args.embed_size < 1
            or args.layers < 1 or args.heads < 1 or args.embed_size % args.heads != 0
            or not 0 <= args.dropout < 1 or not 0 < args.validation_fraction < 1):
        parser.error("Invalid hyperparameters; sizes and learning rate must be positive, "
                     "embed-size must be divisible by heads, and dropout must be in [0, 1)")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    training, validation = split_training_reviews(
        read_reviews(args.data_dir, "train"), args.validation_fraction, args.seed
    )
    vocabulary = build_vocabulary(training, args.vocab_size)
    train_loader = make_loader(training, vocabulary, args.max_length, args.batch_size, shuffle=True)
    validation_loader = make_loader(validation, vocabulary, args.max_length, args.batch_size)
    config = dict(vocab_size=len(vocabulary), embed_size=args.embed_size, num_layers=args.layers,
                  num_heads=args.heads, num_classes=2, max_seq_length=args.max_length,
                  dropout_p=args.dropout, ff_hidden_mult=4)
    model = Transformer(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    print(f"Device: {device}; training: {len(training)}; validation: {len(validation)}; vocabulary: {len(vocabulary)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs, mask=create_padding_mask(inputs))
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            total += labels.numel()
        val_loss, val_accuracy = evaluate(model, validation_loader, device)
        if not torch.isfinite(torch.tensor(val_loss)):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
        print(f"Epoch {epoch}: train loss {total_loss / total:.4f}, "
              f"validation loss {val_loss:.4f}, accuracy {val_accuracy:.2%}")
        if val_loss < best_loss:
            best_loss = val_loss
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "config": config,
                        "vocabulary": vocabulary, "epoch": epoch}, args.checkpoint)
            print(f"Saved best checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
