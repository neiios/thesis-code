# Usage

1. Create a `.env` file in the root of the project based on the example file.

2. Run the following commands:

```sh
# Generate dataset
scala run generate.sc -- \
  --model openai/gpt-4.1 \
  --samples-per-request 20 \
  --snippets-per-topic 40 \
  --topics-file ./topics.json \
  --schema-file ./schema.json \
  --prompts-file ./prompts.yaml \
  --output-dir ./results

# Merged separate datasets
cat ./results/classes_for_data_snippets.jsonl  ./results/null_checks_snippets.jsonl  ./results/throws_snippets.jsonl > ./results/raw_full.jsonl

# Tokenize snippets
scala run ./src/tokenize.sc -- \
  --input ./results/raw_full.jsonl \
  --output-dir ./results
```
