"""Convert the OASST1 flat messages export into tree-disjoint dialogue splits."""

import argparse
import gzip
import json
import random
from collections import Counter
from pathlib import Path


def load_messages(path):
    opener = gzip.open if path.suffix == ".gz" else open
    messages = {}
    with opener(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(message, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            message_id = message.get("message_id")
            tree_id = message.get("message_tree_id")
            parent_id = message.get("parent_id")
            if (not isinstance(message_id, str) or not message_id
                    or not isinstance(tree_id, str) or not tree_id
                    or (parent_id is not None and (not isinstance(parent_id, str) or not parent_id))
                    or message.get("role") not in ("prompter", "assistant")
                    or not isinstance(message.get("deleted"), bool)
                    or message.get("review_result") not in (True, False, None)):
                raise ValueError(f"{path}:{line_number}: invalid OASST message metadata")
            if message_id in messages:
                raise ValueError(f"{path}:{line_number}: duplicate message_id {message_id}")
            messages[message_id] = message
    if not messages:
        raise ValueError(f"No messages found in {path}")
    return messages


def prepare_conversations(messages, language):
    by_tree = {}
    stats = Counter()
    for message in messages.values():
        if message["role"] != "assistant":
            continue
        path = []
        seen = set()
        current = message
        tree_id = message["message_tree_id"]
        while current is not None:
            message_id = current["message_id"]
            if message_id in seen:
                raise ValueError(f"Cycle detected in tree {tree_id}")
            seen.add(message_id)
            if current["message_tree_id"] != tree_id:
                raise ValueError(f"Parent in a different tree for message {message_id}")
            path.append(current)
            parent_id = current.get("parent_id")
            if parent_id is None:
                if message_id != tree_id:
                    raise ValueError(f"Root message does not match tree ID {tree_id}")
                break
            if parent_id not in messages:
                raise ValueError(f"Missing parent {parent_id} for message {message_id}")
            parent = messages[parent_id]
            if parent["role"] == current["role"]:
                raise ValueError(f"Consecutive messages have the same role at {message_id}")
            current = parent
        path.reverse()
        if path[0]["role"] != "prompter":
            raise ValueError(f"Tree {tree_id} does not start with a prompter")
        if any(item["deleted"] or item.get("review_result") is False for item in path):
            stats["excluded_deleted_or_rejected"] += 1
            continue
        if language != "all" and any(item.get("lang") != language for item in path):
            stats["excluded_language"] += 1
            continue
        if any(not isinstance(item.get("text"), str) or not item["text"].strip() for item in path):
            stats["excluded_empty"] += 1
            continue
        conversation = [{"role": "user" if item["role"] == "prompter" else "assistant",
                         "content": item["text"]} for item in path]
        by_tree.setdefault(tree_id, []).append({"messages": conversation, "last_response_only": True})
        stats["included_responses"] += 1
    return by_tree, stats


def write_splits(by_tree, output_dir, seed, validation_fraction, test_fraction):
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1")
    tree_ids = sorted(by_tree)
    if len(tree_ids) < 3:
        raise ValueError("At least three usable conversation trees are needed for train, validation and test")
    random.Random(seed).shuffle(tree_ids)
    val_count = max(1, round(len(tree_ids) * validation_fraction))
    test_count = max(1, round(len(tree_ids) * test_fraction))
    if val_count + test_count >= len(tree_ids):
        raise ValueError("Split fractions leave no trees for training")
    splits = {
        "validation": tree_ids[:val_count],
        "test": tree_ids[val_count:val_count + test_count],
        "train": tree_ids[val_count + test_count:],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / f"{name}.jsonl" for name in splits}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose an empty output directory")
    for name, ids in splits.items():
        count = 0
        with paths[name].open("x", encoding="utf-8") as file:
            for tree_id in ids:
                for conversation in by_tree[tree_id]:
                    file.write(json.dumps(conversation, ensure_ascii=False) + "\n")
                    count += 1
        print(f"{name}: {len(ids)} trees, {count} assistant responses -> {paths[name]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="OASST .messages.jsonl or .jsonl.gz")
    parser.add_argument("--output-dir", type=Path, default=Path("data/oasst"))
    parser.add_argument("--language", default="en", help="Language code (default: en), or all")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.language.strip():
        parser.error("language must not be empty")
    messages = load_messages(args.input)
    by_tree, stats = prepare_conversations(messages, args.language)
    print(f"Read {len(messages)} messages; included {stats['included_responses']} assistant responses; "
          f"excluded paths: deleted/rejected {stats['excluded_deleted_or_rejected']}, "
          f"language {stats['excluded_language']}, empty {stats['excluded_empty']}")
    write_splits(by_tree, args.output_dir, args.seed, args.validation_fraction, args.test_fraction)


if __name__ == "__main__":
    main()
