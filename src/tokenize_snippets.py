import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass, field
from tree_sitter_language_pack import get_parser
import tree_sitter
import sys


@dataclass
class ScalaSnippet:
    original_data: Dict[str, Any]
    ast: Optional[str] = None
    tokens: List[str] = field(default_factory=list)
    contains_errors: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = self.original_data.copy()
        result["ast"] = self.ast
        result["tokens"] = self.tokens
        result["contains_errors"] = self.contains_errors
        return result


def traverse_tree_types(node: tree_sitter.Node) -> List[str]:
    tokens = [node.type]
    for child in node.children:
        tokens.extend(traverse_tree_types(child))
    return tokens


def parse_scala_code(code: str, parser: tree_sitter.Parser) -> Tuple[str, List[str], bool]:
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    return str(root_node), traverse_tree_types(root_node), root_node.has_error


def read_jsonl(file_path: Path) -> Iterator[Tuple[int, Optional[Dict[str, Any]]]]:
    with open(file_path, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file, 1):
            try:
                data = json.loads(line.strip())
                if isinstance(data, dict):
                    yield idx, data
                else:
                    print(f"Line {idx}: Not a valid JSON object")
                    yield idx, None
            except json.JSONDecodeError:
                print(f"Line {idx}: JSON parsing error")
                yield idx, None


def process_snippet(data: Dict[str, Any], parser: tree_sitter.Parser, line_num: int) -> ScalaSnippet:
    snippet = ScalaSnippet(original_data=data)
    entry_id = data.get("id", "N/A")

    code = data.get("code")
    if not code or not isinstance(code, str):
        print(f"Line {line_num} (ID: {entry_id}): No valid code found")
        return snippet

    try:
        ast_str, tokens, has_error = parse_scala_code(code, parser)
        snippet.ast = ast_str
        snippet.tokens = tokens
        snippet.contains_errors = has_error

        if has_error:
            print(f"Line {line_num} (ID: {entry_id}): Syntax errors detected")
    except Exception as e:
        print(f"Line {line_num} (ID: {entry_id}): Parsing error - {e}")

    return snippet


def process_jsonl_dataset(input_path: Path, output_path: Path) -> None:
    if not input_path.exists() or not input_path.is_file():
        print(f"Error: Input file not found or invalid: {input_path}")
        sys.exit(1)

    parser = get_parser("scala")
    print(f"Processing: {input_path} → {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    json_errors = 0
    processing_errors = 0
    syntax_errors = 0

    with open(output_path, "w", encoding="utf-8") as outfile:
        for line_num, data in read_jsonl(input_path):
            if data is None:
                json_errors += 1
                continue

            try:
                snippet = process_snippet(data, parser, line_num)
                outfile.write(json.dumps(snippet.to_dict()) + "\n")

                processed += 1
                if snippet.contains_errors:
                    syntax_errors += 1
            except Exception as e:
                print(f"Line {line_num}: Processing error - {e}")
                processing_errors += 1

    print("\n--- Processing Summary ---")
    print(f"Successfully processed: {processed}")
    print(f"With syntax errors: {syntax_errors}")
    print(f"JSON errors: {json_errors}")
    print(f"Processing errors: {processing_errors}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Scala code snippets and add AST information")
    parser.add_argument("-i", "--input-file", type=Path, required=True, help="Input JSONL file path")
    parser.add_argument("-o", "--output-file", type=Path, required=True, help="Output JSONL file path")

    args = parser.parse_args()
    process_jsonl_dataset(args.input_file, args.output_file)
    print("Done")
