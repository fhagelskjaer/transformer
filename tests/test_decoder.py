import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from test_llm.decoder import (ASSISTANT, END, IGNORE_INDEX, USER, Decoder, DialogueDataset,
                              batch_loss, decode, encode, generate, make_loader, read_conversations)


def conversation(user, assistant):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = Decoder(embed_size=8, num_layers=1, num_heads=2,
                             max_seq_length=16, dropout_p=0)

    def test_causal_attention_and_padding(self):
        self.model.eval()
        left = torch.tensor([[USER, 69, ASSISTANT, 70, 71]])
        changed_future = torch.tensor([[USER, 69, ASSISTANT, 100, 101]])
        padded = torch.tensor([[USER, 69, ASSISTANT, 70, 71, 0, 0]])
        with torch.no_grad():
            original = self.model(left)
            changed = self.model(changed_future)
            padded_logits = self.model(padded)
        torch.testing.assert_close(original[:, :3], changed[:, :3])
        torch.testing.assert_close(original, padded_logits[:, :5], rtol=1e-5, atol=1e-6)

    def test_assistant_targets_and_multiturn_context(self):
        messages = conversation("a", "b") + conversation("c", "d")
        dataset = DialogueDataset([messages], 16)
        self.assertEqual(len(dataset), 2)
        inputs, labels = dataset[0]
        self.assertEqual(inputs, [USER, *encode("a"), END, ASSISTANT, *encode("b")])
        self.assertEqual(labels, [IGNORE_INDEX] * 3 + encode("b") + [END])
        self.assertEqual(dataset[1][0][:len(inputs)], inputs)
        self.assertEqual(decode(encode("Héllo 🌍")), "Héllo 🌍")

    def test_truncation_does_not_add_false_end(self):
        dataset = DialogueDataset([conversation("x", "abcdef")], 3)
        inputs, targets = dataset[0]
        self.assertEqual(inputs[0], ASSISTANT)
        self.assertEqual(targets, encode("abc"))
        self.assertEqual(dataset.truncated_responses, 1)

    def test_model_learns_tiny_batch(self):
        loader = make_loader([conversation("hi", "ok")], 16, 1)
        inputs, labels = next(iter(loader))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        initial = batch_loss(self.model, inputs, labels).item()
        for _ in range(35):
            optimizer.zero_grad()
            loss = batch_loss(self.model, inputs, labels)
            loss.backward()
            optimizer.step()
        self.assertLess(batch_loss(self.model, inputs, labels).item(), initial * 0.3)

    def test_generation_stops_on_end(self):
        class EndModel(torch.nn.Module):
            max_seq_length = 4

            def forward(self, tokens):
                result = torch.zeros((1, tokens.size(1), 260))
                result[:, :, END] = 100
                return result

        result, ended = generate(EndModel(), [USER, *encode("hello"), END, ASSISTANT],
                                 torch.device("cpu"), temperature=0)
        self.assertEqual(result, [])
        self.assertTrue(ended)

    def test_generation_respects_limit_and_tiny_temperature(self):
        class ByteModel(torch.nn.Module):
            max_seq_length = 4

            def forward(self, tokens):
                result = torch.zeros((1, tokens.size(1), 260))
                result[:, :, encode("a")[0]] = 100
                return result

        result, ended = generate(ByteModel(), [USER, *encode("hello"), END, ASSISTANT],
                                 torch.device("cpu"), max_new_tokens=3, temperature=1e-300)
        self.assertEqual(decode(result), "aaa")
        self.assertFalse(ended)

    def test_invalid_data_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text('{"messages":[{"role":"assistant","content":"oops"}]}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "user/assistant pairs"):
                read_conversations(path)


class DecoderCommandLineTests(unittest.TestCase):
    def test_train_evaluate_and_chat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("train", "val", "test"):
                (root / f"{name}.jsonl").write_text(
                    json.dumps({"messages": conversation("hi", "hello")}) + "\n", encoding="utf-8")
            checkpoint = root / "decoder.pt"
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

            def run(args, input_text=None):
                return subprocess.run([sys.executable, "-m", *args], input=input_text,
                                      check=True, capture_output=True, text=True, env=env).stdout

            trained = run(["test_llm.train_decoder", "--train-data", str(root / "train.jsonl"),
                           "--validation-data", str(root / "val.jsonl"), "--checkpoint", str(checkpoint),
                           "--epochs", "1", "--batch-size", "1", "--max-length", "16",
                           "--embed-size", "8", "--layers", "1", "--heads", "2", "--device", "cpu"])
            self.assertIn("Saved best checkpoint", trained)
            result = run(["test_llm.test_decoder", "--data", str(root / "test.jsonl"),
                          "--checkpoint", str(checkpoint), "--device", "cpu"])
            self.assertIn("Test responses: 1", result)
            result = run(["test_llm.chat", "--checkpoint", str(checkpoint),
                          "--max-new-tokens", "3", "--temperature", "0", "--device", "cpu"],
                         "hi\n/reset\n/quit\n")
            self.assertIn("Bot:", result)
            self.assertIn("Conversation reset.", result)


if __name__ == "__main__":
    unittest.main()
