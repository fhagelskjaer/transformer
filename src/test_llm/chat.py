"""Chat interactively with a trained decoder."""

import argparse
import math
from pathlib import Path

from .decoder import ASSISTANT, END, USER, decode, encode, generate, load_model, resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/decoder.pt"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 selects greedy decoding")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if (args.max_new_tokens < 1 or not math.isfinite(args.temperature)
            or args.temperature < 0 or not 1 <= args.top_k <= 257):
        parser.error("max-new-tokens must be positive, temperature non-negative, and top-k between 1 and 257")
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    history = []
    print("Chat ready. Type /reset to clear history or /quit to exit.")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text == "/quit":
            break
        if text == "/reset":
            history = []
            print("Conversation reset.")
            continue
        if not text:
            continue
        prompt = history + [USER, *encode(text), END, ASSISTANT]
        response, ended = generate(model, prompt, device, args.max_new_tokens,
                                   args.temperature, args.top_k)
        rendered = decode(response)
        safe_text = "".join(char if char.isprintable() or char in "\n\t" else repr(char)[1:-1]
                            for char in rendered)
        print(f"Bot: {safe_text}")
        if not ended:
            print("(Response stopped at the token limit.)")
        history = (prompt + response + [END])[-model.max_seq_length:]


if __name__ == "__main__":
    main()
