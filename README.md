# Usage

1. Create a `.env` file in the root of the project based on the example file.

2. Run the following commands:

```sh
# Generate dataset
scala run ./src/generate.sc -- \
  --model openai/gpt-4.1 \
  --samples-per-request 20 \
  --snippets-per-topic 40 \
  --topics-file ./src/topics.json \
  --schema-file ./src/schema.json \
  --prompts-file ./src/prompts.yaml \
  --output-dir ./results

# Merged separate datasets
cat ./results/classes_for_data_snippets.jsonl  ./results/null_checks_snippets.jsonl  ./results/throws_snippets.jsonl > ./results/raw_full.jsonl

# Tokenize snippets
scala run ./src/tokenize.sc -- \
  --input ./results/raw_full.jsonl \
  --output-dir ./results

# Optimize CNN
uv run -m src.cnn.optimize -i results/processed_full.jsonl -o results/lstm_optimized --max-trials 20 --overwrite

# Optimize LSTM
uv run -m src.lstm.optimize -i results/processed_full.jsonl -o results/lstm_optimized --max-trials 20 --overwrite
```
