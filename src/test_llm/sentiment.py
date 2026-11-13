"""Shared IMDb data preparation and evaluation helpers."""

import random
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .transformers import create_padding_mask

PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"
LABELS = ("neg", "pos")


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def read_reviews(root, split):
    reviews = []
    for label, name in enumerate(LABELS):
        directory = Path(root) / split / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing review directory: {directory}")
        files = sorted(directory.glob("*.txt"))
        if not files:
            raise ValueError(f"No .txt reviews found in {directory}")
        reviews.extend((path.read_text(encoding="utf-8"), label) for path in files)
    return reviews


def split_training_reviews(reviews, validation_fraction, seed):
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1")
    rng = random.Random(seed)
    training, validation = [], []
    for label in range(len(LABELS)):
        group = [review for review in reviews if review[1] == label]
        if len(group) < 2:
            raise ValueError("At least two reviews per class are needed for a validation split")
        rng.shuffle(group)
        count = min(len(group) - 1, max(1, round(len(group) * validation_fraction)))
        validation.extend(group[:count])
        training.extend(group[count:])
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation


def build_vocabulary(reviews, max_vocab_size):
    if max_vocab_size < 3:
        raise ValueError("Vocabulary size must be at least 3")
    counts = Counter(token for text, _ in reviews for token in tokenize(text))
    # Alphabetical tie-breaking keeps vocabulary construction reproducible.
    words = sorted(counts, key=lambda word: (-counts[word], word))
    return [PAD, UNK, CLS] + words[:max_vocab_size - 3]


class ReviewDataset(Dataset):
    def __init__(self, reviews, vocabulary, max_length):
        if max_length < 2:
            raise ValueError("Maximum length must be at least 2")
        indices = {token: index for index, token in enumerate(vocabulary)}
        self.samples = [
            ([indices[CLS]] + [indices.get(word, indices[UNK])
                              for word in tokenize(text)[:max_length - 1]], label)
            for text, label in reviews
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_reviews(batch):
    sequences, labels = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(sequence, dtype=torch.long) for sequence in sequences],
        batch_first=True, padding_value=0,
    )
    return inputs, torch.tensor(labels, dtype=torch.long)


def make_loader(reviews, vocabulary, max_length, batch_size, shuffle=False):
    return DataLoader(
        ReviewDataset(reviews, vocabulary, max_length),
        batch_size=batch_size, shuffle=shuffle, collate_fn=collate_reviews,
    )


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs, mask=create_padding_mask(inputs))
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    if not total:
        raise ValueError("Cannot evaluate an empty dataset")
    return total_loss / total, correct / total
