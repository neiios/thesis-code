from dataclasses import dataclass
import json
from pathlib import Path
import keras
import sklearn.preprocessing
import numpy as np

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"


@dataclass
class Entry:
    id: str
    timestamp: int
    topicUsed: str
    label: str  # either vulnerability category class or "idiomatic"; transformed category + isIdiomatic object fields
    code: str
    tokens: list[str]


def read_dataset(jsonl_path: Path) -> list[Entry]:
    items: list[Entry] = []
    error_count = 0

    with jsonl_path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            try:
                obj = json.loads(line)

                if obj["isIdiomatic"]:
                    label = "idiomatic"
                else:
                    label = obj["category"]

                item = Entry(
                    id=obj["id"],
                    timestamp=obj["timestamp"],
                    topicUsed=obj["topicUsed"],
                    label=label,
                    code=obj["code"],
                    tokens=obj["tokens"],
                )
                items.append(item)

            except (KeyError, json.JSONDecodeError) as e:
                error_count += 1
                print(f"[Error] Line {i}: Failed to parse - {str(e)}")

    print(
        f"[Load] Parsed {len(items):,} LineItem objects from {jsonl_path} ({error_count} errors)"
    )
    return items


def create_vocabulary(data_path: Path) -> dict[str, int]:
    token_set: set[str] = set()

    with data_path.open(encoding="utf-8") as fh:
        for line in fh:
            token_set.update(json.loads(line).get("tokens", []))

    tok2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    tok2id.update({tok: i + 2 for i, tok in enumerate(sorted(token_set))})

    print(f"[Vocabulary] Total unique tokens: {len(tok2id):,}")
    return tok2id


def save_vocabulary(vocabulary: dict[str, int], outFile: Path) -> None:
    outFile.write_text(json.dumps(vocabulary, indent=2, ensure_ascii=False))
    print(f"[Vocabulary] Saved → {outFile}")


def preprocess_data(
    entries: list[Entry], vocabulary: dict[str, int], max_seq_length: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    unk = vocabulary[UNK_TOKEN]
    seq_ids = [[vocabulary.get(tok, unk) for tok in entry.tokens] for entry in entries]

    X = keras.utils.pad_sequences(
        seq_ids,
        maxlen=max_seq_length,
        padding="post",
        truncating="post",
        value=0,
        dtype="int32",
    )

    labels = [entry.label for entry in entries]
    le = sklearn.preprocessing.LabelEncoder().fit(labels)
    indices = np.asarray(le.transform(labels), dtype=np.int64)
    y = np.eye(len(le.classes_), dtype=np.float32)[indices]

    print(f"[Data] X{X.shape}, y{y.shape} - classes: {list(le.classes_)}")
    return X, y, list(le.classes_)
