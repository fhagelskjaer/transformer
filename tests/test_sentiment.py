import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from test_llm.sentiment import build_vocabulary, make_loader, split_training_reviews
from test_llm.transformers import Transformer, create_padding_mask


class TransformerTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = Transformer(12, 8, 1, 2, 2, max_seq_length=8, dropout_p=0.1)
        self.inputs = torch.tensor([[2, 3, 4], [2, 5, 0]])
        self.mask = create_padding_mask(self.inputs)

    def test_registered_parameters_train_and_state_round_trips(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        before = {name: param.detach().clone() for name, param in self.model.named_parameters()}
        loss = torch.nn.functional.cross_entropy(self.model(self.inputs, self.mask), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(all(param.grad is not None for param in self.model.parameters()))
        self.assertTrue(all(torch.isfinite(param.grad).all() for param in self.model.parameters()))
        optimizer.step()
        after = dict(self.model.named_parameters())
        for name in ("encoder.token_embedding.weight", "encoder.layers.0.attention.query.weight",
                     "encoder.layers.0.ff.fc1.weight", "encoder.layers.0.norm1.gamma",
                     "encoder.norm.gamma", "classifier.weight"):
            self.assertFalse(torch.equal(before[name], after[name]), name)

        self.model.eval()
        expected = self.model(self.inputs, self.mask)
        self.assertTrue(torch.equal(expected, self.model(self.inputs, self.mask)))
        other = Transformer(12, 8, 1, 2, 2, max_seq_length=8, dropout_p=0.1)
        other.load_state_dict(self.model.state_dict())
        other.eval()
        torch.testing.assert_close(expected, other(self.inputs, self.mask))

    def test_padding_does_not_change_prediction(self):
        self.model.eval()
        padded = torch.tensor([[2, 3, 4, 0, 0]])
        with torch.no_grad():
            expected = self.model(self.inputs[:1], create_padding_mask(self.inputs[:1]))
            actual = self.model(padded, create_padding_mask(padded))
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)

    def test_learns_a_tiny_batch(self):
        labels = torch.tensor([0, 1])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.model.eval()
        initial = torch.nn.functional.cross_entropy(self.model(self.inputs, self.mask), labels).item()
        for _ in range(20):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.model(self.inputs, self.mask), labels)
            loss.backward()
            optimizer.step()
        final = torch.nn.functional.cross_entropy(self.model(self.inputs, self.mask), labels).item()
        self.assertLess(final, initial * 0.25)

    def test_training_split_vocabulary_and_unknown_words(self):
        reviews = [("negative one", 0), ("negative two", 0),
                   ("positive one", 1), ("positive two", 1)]
        training, validation = split_training_reviews(reviews, 0.5, 42)
        self.assertEqual(len(training), 2)
        self.assertEqual(len(validation), 2)
        self.assertEqual({label for _, label in training}, {0, 1})
        vocabulary = build_vocabulary(training, 20)
        inputs, _ = next(iter(make_loader([("unseen", 0)], vocabulary, 8, 1)))
        self.assertEqual(inputs.tolist(), [[2, 1]])


class CommandLineTests(unittest.TestCase):
    def test_train_save_and_run_both_test_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "aclImdb"
            for split in ("train", "test"):
                for label in ("neg", "pos"):
                    directory = root / split / label
                    directory.mkdir(parents=True)
                    for index in range(3):
                        (directory / f"{index}.txt").write_text(
                            f"{'terrible dull' if label == 'neg' else 'wonderful fun'} film {index}",
                            encoding="utf-8",
                        )
            checkpoint = Path(tmp) / "model.pt"
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

            def run(*args):
                return subprocess.run([sys.executable, "-m", *args], check=True,
                                      capture_output=True, text=True, env=env).stdout

            train_output = run("test_llm.train", "--data-dir", str(root), "--checkpoint", str(checkpoint),
                               "--epochs", "1", "--batch-size", "2", "--max-length", "8",
                               "--embed-size", "8", "--layers", "1", "--heads", "2", "--device", "cpu")
            self.assertIn("Saved best checkpoint", train_output)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["epoch"], 1)
            self.assertEqual(saved["config"]["vocab_size"], len(saved["vocabulary"]))
            output = run("test_llm.test_model", "--checkpoint", str(checkpoint),
                         "--data-dir", str(root), "--device", "cpu")
            self.assertIn("Test reviews: 6", output)
            self.assertIn("accuracy:", output)
            prediction = run("test_llm.test_model", "--checkpoint", str(checkpoint),
                             "--text", "wonderful film", "--device", "cpu")
            self.assertIn("Prediction:", prediction)


if __name__ == "__main__":
    unittest.main()
