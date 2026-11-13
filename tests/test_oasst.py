import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_llm.decoder import ASSISTANT, END, IGNORE_INDEX, DialogueDataset, encode, read_conversations
from test_llm.prepare_oasst import load_messages, prepare_conversations, write_splits


def message(message_id, tree, parent, role, text, **kwargs):
    return {"message_id": message_id, "message_tree_id": tree, "parent_id": parent,
            "role": role, "text": text, "lang": "en", "deleted": False,
            "review_result": True, **kwargs}


def sample_messages():
    rows = [
        message("root1", "root1", None, "prompter", "question"),
        message("a1", "root1", "root1", "assistant", "first"),
        message("u2", "root1", "a1", "prompter", "followup"),
        message("a2", "root1", "u2", "assistant", "second"),
        message("a3", "root1", "root1", "assistant", "alternative"),
        message("rejected", "root1", "root1", "assistant", "bad", review_result=False),
        message("descendant", "root1", "rejected", "prompter", "next"),
        message("desc-answer", "root1", "descendant", "assistant", "ignored"),
        message("root2", "root2", None, "prompter", "two"),
        message("b1", "root2", "root2", "assistant", "answer"),
        message("root3", "root3", None, "prompter", "three"),
        message("c1", "root3", "root3", "assistant", "answer"),
        message("root4", "root4", None, "prompter", "cuatro", lang="es"),
        message("d1", "root4", "root4", "assistant", "respuesta", lang="es"),
    ]
    return {row["message_id"]: row for row in rows}


class OasstTests(unittest.TestCase):
    def test_branches_filters_and_last_response_targets(self):
        trees, stats = prepare_conversations(sample_messages(), "en")
        self.assertEqual(len(trees["root1"]), 3)
        self.assertEqual(stats["excluded_deleted_or_rejected"], 2)
        self.assertEqual(stats["excluded_language"], 1)
        long_path = next(item for item in trees["root1"] if len(item["messages"]) == 4)
        self.assertEqual([x["content"] for x in long_path["messages"]],
                         ["question", "first", "followup", "second"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(json.dumps(long_path), encoding="utf-8")
            data = DialogueDataset(read_conversations(path), 128)
        self.assertEqual(len(data), 1)
        _, labels = data[0]
        self.assertEqual([token for token in labels if token != IGNORE_INDEX], encode("second") + [END])

    def test_long_response_retains_prompt_context(self):
        conversations = [[{"role": "user", "content": "question"},
                          {"role": "assistant", "content": "x" * 100}]]
        inputs, labels = DialogueDataset(conversations, 16)[0]
        marker = inputs.index(ASSISTANT)
        self.assertGreaterEqual(marker, 7)
        self.assertEqual(len(inputs), 16)
        self.assertNotIn(END, labels)

    def test_missing_parent_and_cycle_fail(self):
        messages = sample_messages()
        messages["a1"]["parent_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "Missing parent"):
            prepare_conversations(messages, "en")
        messages = sample_messages()
        messages["a1"]["parent_id"] = "u2"
        with self.assertRaisesRegex(ValueError, "Cycle"):
            prepare_conversations(messages, "en")

    def test_gzip_and_disjoint_reproducible_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "messages.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as file:
                for item in sample_messages().values():
                    file.write(json.dumps(item) + "\n")
            trees, _ = prepare_conversations(load_messages(source), "en")
            write_splits(trees, root / "out1", 42, 0.1, 0.1)
            write_splits(trees, root / "out2", 42, 0.1, 0.1)
            roots = []
            for name in ("train", "validation", "test"):
                first = (root / "out1" / f"{name}.jsonl").read_text()
                self.assertEqual(first, (root / "out2" / f"{name}.jsonl").read_text())
                roots.append({json.loads(line)["messages"][0]["content"] for line in first.splitlines()})
            self.assertTrue(all(not roots[i] & roots[j] for i in range(3) for j in range(i)))
            with self.assertRaises(FileExistsError):
                write_splits(trees, root / "out1", 42, 0.1, 0.1)

    def test_conversion_and_training_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "messages.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in sample_messages().values()))
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            def run(*args):
                return subprocess.run([sys.executable, "-m", *args], env=env, text=True,
                                      capture_output=True, check=True).stdout
            run("test_llm.prepare_oasst", "--input", str(path), "--output-dir", str(root / "out"))
            output = run("test_llm.train_decoder", "--train-data", str(root / "out/train.jsonl"),
                         "--validation-data", str(root / "out/validation.jsonl"),
                         "--checkpoint", str(root / "model.pt"), "--epochs", "1",
                         "--embed-size", "8", "--layers", "1", "--heads", "2",
                         "--max-length", "16", "--batch-size", "2", "--device", "cpu")
            self.assertIn("Saved best checkpoint", output)
            output = run("test_llm.test_decoder", "--data", str(root / "out/test.jsonl"),
                         "--checkpoint", str(root / "model.pt"), "--device", "cpu")
            self.assertIn("Test responses:", output)


if __name__ == "__main__":
    unittest.main()
