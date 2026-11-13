"""A byte-level, decoder-only dialogue model and shared data utilities."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .transformers import Linear, TransformerEncoder, create_causal_mask, create_padding_mask

PAD, USER, ASSISTANT, END = range(4)
BYTE_OFFSET = 4
VOCAB_SIZE = BYTE_OFFSET + 256
IGNORE_INDEX = -100
FORMAT_VERSION = 1


@dataclass
class Conversation:
    messages: list
    last_response_only: bool = False


def encode(text):
    return [BYTE_OFFSET + byte for byte in text.encode("utf-8")]


def decode(tokens):
    return bytes(token - BYTE_OFFSET for token in tokens).decode("utf-8", errors="replace")


class Decoder(nn.Module):
    """Causal language model using the existing transformer block stack."""

    def __init__(self, embed_size=64, num_layers=2, num_heads=4,
                 max_seq_length=128, dropout_p=0.1, ff_hidden_mult=4):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.backbone = TransformerEncoder(
            VOCAB_SIZE, embed_size, num_layers, num_heads,
            max_seq_length, dropout_p, ff_hidden_mult,
        )
        self.output = Linear(embed_size, VOCAB_SIZE)

    def forward(self, tokens):
        causal_mask = create_causal_mask(tokens.size(1)).to(device=tokens.device, dtype=torch.bool)
        mask = causal_mask & create_padding_mask(tokens, PAD)
        hidden = self.backbone(tokens, mask=mask, training=self.training)
        return self.output(hidden)


def read_conversations(path):
    conversations = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            messages = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(messages, list) or not messages or len(messages) % 2:
                raise ValueError(f"{path}:{line_number}: messages must contain user/assistant pairs")
            last_response_only = record.get("last_response_only", False)
            if not isinstance(last_response_only, bool):
                raise ValueError(f"{path}:{line_number}: last_response_only must be a boolean")
            for index, message in enumerate(messages):
                role = "user" if index % 2 == 0 else "assistant"
                if (not isinstance(message, dict) or message.get("role") != role
                        or not isinstance(message.get("content"), str)
                        or not message["content"].strip()):
                    raise ValueError(f"{path}:{line_number}: message {index + 1} must be a non-empty {role} message")
            conversations.append(Conversation(messages, last_response_only))
    if not conversations:
        raise ValueError(f"No conversations found in {path}")
    return conversations


class DialogueDataset(Dataset):
    def __init__(self, conversations, max_length):
        if max_length < 2:
            raise ValueError("Maximum length must be at least 2")
        self.samples = []
        self.truncated_responses = 0
        for conversation in conversations:
            if isinstance(conversation, Conversation):
                messages = conversation.messages
                last_response_only = conversation.last_response_only
            else:
                messages = conversation
                last_response_only = False
            history = []
            for index in range(0, len(messages), 2):
                history += [USER, *encode(messages[index]["content"]), END, ASSISTANT]
                response = encode(messages[index + 1]["content"])
                if last_response_only and index + 2 < len(messages):
                    history += response + [END]
                    continue
                target = response + [END]
                # Reserve room for recent prompt context even for long answers.
                context_budget = min(len(history), max(1, max_length // 2))
                target_limit = max_length + 1 - context_budget
                if len(target) > target_limit:
                    target = target[:target_limit]
                    self.truncated_responses += 1
                # Preserve the assistant marker while retaining as much history as fits.
                context = history[-(max_length + 1 - len(target)):]
                sequence = context + target
                inputs = sequence[:-1]
                labels = [IGNORE_INDEX] * (len(context) - 1) + target
                self.samples.append((inputs, labels))
                history += response + [END]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_dialogues(batch):
    inputs, labels = zip(*batch)
    return (
        nn.utils.rnn.pad_sequence([torch.tensor(item, dtype=torch.long) for item in inputs],
                                  batch_first=True, padding_value=PAD),
        nn.utils.rnn.pad_sequence([torch.tensor(item, dtype=torch.long) for item in labels],
                                  batch_first=True, padding_value=IGNORE_INDEX),
    )


def make_loader(conversations, max_length, batch_size, shuffle=False):
    return DataLoader(DialogueDataset(conversations, max_length), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_dialogues)


def batch_loss(model, inputs, labels):
    logits = model(inputs)
    return nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1),
                                       ignore_index=IGNORE_INDEX, reduction="sum")


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens, correct = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1),
                                               ignore_index=IGNORE_INDEX, reduction="sum")
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite evaluation loss")
            mask = labels != IGNORE_INDEX
            total_loss += loss.item()
            total_tokens += mask.sum().item()
            correct += ((logits.argmax(dim=-1) == labels) & mask).sum().item()
    if not total_tokens:
        raise ValueError("No assistant target tokens to evaluate")
    return total_loss / total_tokens, correct / total_tokens


def perplexity(loss):
    return math.exp(loss) if loss < 709 else float("inf")


def resolve_device(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != "byte-dialogue" or checkpoint.get("version") != FORMAT_VERSION:
        raise ValueError("Unsupported decoder checkpoint format")
    model = Decoder(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def generate(model, prompt, device, max_new_tokens=128, temperature=0.8, top_k=40):
    if not prompt or prompt[-1] != ASSISTANT:
        raise ValueError("Prompt must end with an assistant marker")
    if max_new_tokens < 1 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("max_new_tokens must be positive and temperature must be finite and non-negative")
    if not 1 <= top_k <= VOCAB_SIZE - 3:
        raise ValueError(f"top_k must be between 1 and {VOCAB_SIZE - 3}")
    model.eval()
    tokens = list(prompt)
    generated = []
    for _ in range(max_new_tokens):
        inputs = torch.tensor([tokens[-model.max_seq_length:]], dtype=torch.long, device=device)
        logits = model(inputs)[0, -1].clone()
        logits[[PAD, USER, ASSISTANT]] = -torch.inf
        if temperature == 0:
            next_token = logits.argmax().item()
        else:
            values, indices = logits.topk(top_k)
            probabilities = torch.softmax((values.double() - values.max().double()) / temperature, dim=-1)
            next_token = indices[torch.multinomial(probabilities, 1).item()].item()
        if next_token == END:
            return generated, True
        generated.append(next_token)
        tokens.append(next_token)
    return generated, False
