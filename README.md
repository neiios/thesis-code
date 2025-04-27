# Usage

1. Create a `.env` file in the root of the project based on the example file.

2. Run the following commands:

```sh
# Generate dataset
pushd generator && scala run generateDataset.scala && popd

# Tokenize snippets
pushd ast-tokenizer && uv run tokenize_snippets.py \
  -i ../dataset/raw_full.jsonl \
  -o ../dataset/tokenized_full.jsonl && popd

# Create vocabulary
pushd vocab && uv run create_vocab.py \
  -i ../dataset/tokenized_full.jsonl \
  -o ../dataset/vocab.json && popd

# Train LSTM
pushd lstm && python train_lstm.py \
  -i ../results/tokenized_full.jsonl \
  -v ../results/vocab.json \
  -o ./trained_model && popd
```
