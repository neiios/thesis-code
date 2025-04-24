import json
import argparse
from pathlib import Path
import sys
from typing import Set, Dict, Any, Iterator, Tuple


def read_jsonl(file_path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open(file_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, 1):
            try:
                data = json.loads(line.strip())
                if isinstance(data, dict):
                    yield i, data
                else:
                    print(f"Line {i}: Not a JSON object")
            except json.JSONDecodeError:
                print(f"Line {i}: JSON parsing error")


def should_process(data: Dict[str, Any], line_num: int) -> bool:
    if data.get("contains_errors", False):
        return False

    tokens = data.get("tokens")
    if not tokens or not isinstance(tokens, list):
        entry_id = data.get("id", "N/A")
        print(f"Line {line_num} (ID: {entry_id}): Missing or invalid tokens")
        return False

    return True


def save_vocabulary(vocab: Set[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_vocab = sorted(list(vocab))

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(sorted_vocab, file, indent=2)


def extract_tokens_and_build_vocab(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Processing: {input_path} → {output_path}")

    vocabulary: Set[str] = set()
    processed = 0
    skipped = 0

    for line_num, data in read_jsonl(input_path):
        if not should_process(data, line_num):
            skipped += 1
            continue

        tokens = data["tokens"]
        vocabulary.update(t for t in tokens if isinstance(t, str))
        processed += 1

    if not vocabulary:
        print("Warning: No tokens extracted")
    else:
        print(f"Found {len(vocabulary)} unique tokens")
        save_vocabulary(vocabulary, output_path)

    print("\n--- Summary ---")
    print(f"Processed lines: {processed}")
    print(f"Skipped lines: {skipped}")
    print(f"Unique tokens: {len(vocabulary)}")
    print(f"Vocabulary saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract tokens from a JSONL file and create a vocabulary"
    )
    parser.add_argument(
        "-i", "--input-file", type=Path, required=True, help="Input JSONL file path"
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        required=True,
        help="Output JSON vocabulary file path",
    )

    args = parser.parse_args()
    extract_tokens_and_build_vocab(args.input_file, args.output_file)
