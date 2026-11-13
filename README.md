# Train the transformer on IMDb sentiment

The model in `src/test_llm/transformers.py` is an encoder-only **classifier**,
not an autoregressive text generator. A suitable first dataset is Stanford's
[Large Movie Review Dataset (IMDb)](https://ai.stanford.edu/~amaas/data/sentiment/):
50,000 labeled English movie reviews, with separate 25,000-review training and
test sets. Check the dataset's terms before use.

Install the project's dependencies with `uv sync`, then download and extract
the dataset from Stanford:

```sh
mkdir -p data
curl -L --fail -o data/aclImdb_v1.tar.gz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf data/aclImdb_v1.tar.gz -C data
```

The resulting `data/aclImdb/train/{neg,pos}` and
`data/aclImdb/test/{neg,pos}` directories contain the labeled `.txt` files.
The unlabeled reviews are not used.

## Train

```sh
uv run python -m test_llm.train --data-dir data/aclImdb
```

The script splits the original training set by class (90% training, 10%
validation), builds its word vocabulary **only from the training portion**,
and saves the model with the lowest validation loss to `checkpoints/imdb.pt`.
The checkpoint includes the model parameters, architecture, and vocabulary
needed for inference. It is not a resumable optimizer checkpoint. The official
test set is never used for training or validation.

The defaults use a small model and truncate reviews to the first 127 tokens
plus a classification token; results are not comparable to full-review
benchmarks. Training from scratch can take time, particularly on CPU. CUDA is
selected automatically if available; pass `--device cpu` to force CPU. For a
smaller or longer experiment, for example:

```sh
uv run python -m test_llm.train --data-dir data/aclImdb --epochs 3 --batch-size 16 --max-length 64
```

Use `uv run python -m test_llm.train --help` for all options.

## Test or classify a review

```sh
uv run python -m test_llm.test_model --checkpoint checkpoints/imdb.pt --data-dir data/aclImdb
uv run python -m test_llm.test_model --checkpoint checkpoints/imdb.pt --text "A wonderful and entertaining film."
```

The first command reports loss and accuracy on the official held-out test
set; the second prints the predicted sentiment and class probabilities.
Evaluate the test set after choosing hyperparameters, rather than tuning on it.
Only load checkpoints from sources you trust.

Run the small automated tests without downloading IMDb:

```sh
uv run python -m unittest discover -s tests
```

# Decoder-only chatbot

The separate decoder reuses the transformer blocks with causal attention and a
next-token prediction head. It is an educational model trained from scratch,
not a pretrained assistant. IMDb is for the classifier above, not for chat.

## Conversation data

Provide three separate UTF-8 JSONL files (one JSON object per line): training,
validation, and held-out test. Each object contains alternating non-empty
`user` and `assistant` messages, beginning with a user and ending with an
assistant. A conversation may have multiple turns:

```json
{"messages":[{"role":"user","content":"Hello!"},{"role":"assistant","content":"Hi!"},{"role":"user","content":"How are you?"},{"role":"assistant","content":"I am fine."}]}
```

For a quick smoke test, tiny example files are included in `examples/`. They
are intentionally much too small for a useful chatbot; their test set is
illustrative, not a meaningful benchmark. For a larger experiment, obtain a
dialogue dataset, check its license and usage restrictions, convert it into the
format above, and split by **conversation** before training. Avoid placing the
same conversation in more than one split.

## Train, test, and chat

```sh
uv sync
uv run python -m test_llm.train_decoder \
  --train-data examples/dialogue_train.jsonl \
  --validation-data examples/dialogue_validation.jsonl
uv run python -m test_llm.test_decoder --data examples/dialogue_test.jsonl
uv run python -m test_llm.chat
```

The default checkpoint is `checkpoints/decoder.pt`. Training selects the
checkpoint with the lowest validation loss; it contains the model configuration
and weights, but not optimizer state for resuming. Use `--checkpoint` to choose
another path, and `--device cpu` to force CPU instead of automatic CUDA
selection. Run each module with `--help` for its options.

The tokenizer represents UTF-8 bytes directly (256 byte values plus four role,
end and padding tokens), so there is no vocabulary to download or build.
Training predicts only assistant response bytes and end markers; previous turns
are context. Context is truncated from the left to fit `--max-length` (128
bytes/tokens by default); long responses are also truncated to reserve room
for recent context, without teaching a false end marker. Their count is reported.
Evaluation uses the same
teacher-forced loss and reports byte-token accuracy and perplexity; those
numbers do not measure conversational quality.

Chat keeps recent context, stopping a response at an end marker or
`--max-new-tokens` (128 by default). `/reset` clears context, `/quit` exits,
and `--temperature 0` selects greedy decoding. Other temperatures use top-k
sampling; generated invalid or incomplete UTF-8 bytes are displayed as
replacement characters. Output quality will be poor without sufficient
training data and compute. Load only checkpoints from sources you trust.

## Training with OASST1

The [OpenAssistant Conversations (OASST1) dataset](https://huggingface.co/datasets/OpenAssistant/oasst1)
includes the flat `2023-04-12_oasst_all.messages.jsonl.gz` export. The
converter also accepts the extracted `.jsonl` file; it does not require the
Hugging Face `datasets` package. For the copy under `data/archive/`:

```sh
uv run python -m test_llm.prepare_oasst \
  --input data/archive/2023-04-12_oasst_all.messages.jsonl \
  --output-dir data/oasst

uv run python -m test_llm.train_decoder \
  --train-data data/oasst/train.jsonl \
  --validation-data data/oasst/validation.jsonl \
  --checkpoint checkpoints/oasst-decoder.pt

uv run python -m test_llm.test_decoder \
  --data data/oasst/test.jsonl --checkpoint checkpoints/oasst-decoder.pt
uv run python -m test_llm.chat --checkpoint checkpoints/oasst-decoder.pt
```

Change `--input` to the location of your download if needed. The converter
refuses to overwrite existing output files. It groups by `message_tree_id`,
then splits eligible trees deterministically (80% training, 10% validation,
10% test by default; see `--help`). Consequently branches from the same tree
cannot cross splits. These are custom splits, not the official OASST1
Hugging Face splits.

By default, every message in a path must be tagged English (`--language en`);
use another language code or `--language all` to change this. Paths containing
deleted messages, negatively reviewed messages, or empty text are excluded,
including their descendants. Other tree states, unrated messages, and synthetic
responses are included. Each remaining assistant response is a training example
once, with its path from the root as context; the loss applies only to the
final assistant response. This includes alternative responses without repeatedly
training on their shared earlier responses.

OASST1 is distributed under Apache 2.0; consult the dataset card and license
before use. The *all* export includes unfinished, low-quality, and moderator-
halted trees, and the filters above are **not a safety or privacy filter**.
Review the data and additional content/PII filtering requirements for your use
case; generated text is not moderated.

The byte-level model's default 128-token window is very short for OASST.
When a response would fill the window, training reserves approximately half of
it for recent context and truncates the response without an end marker.
Increase `--max-length` if your hardware permits, but attention memory and
compute grow quadratically with sequence length. Expect long training times
and limited quality from this small model; compare test metrics only after
choosing your hyperparameters on validation data.