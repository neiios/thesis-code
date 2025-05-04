import json
from pathlib import Path
from typing import Dict, List, Tuple
import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_SEQ_LENGTH = 500


def create_vocabulary(data_path: Path) -> Dict[str, int]:
    print(f"Creating vocabulary from: {data_path}")
    token_set = set()

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tokens = data.get("tokens", [])
            if tokens:
                token_set.update(tokens)

    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, token in enumerate(sorted(token_set)):
        token_to_id[token] = i + 2

    print(f"Vocabulary size: {len(token_to_id)}")
    return token_to_id


def preprocess_data(
    sequences: List[List[str]],
    labels: List[str],
    is_idiomatic: List[bool],
    token_to_id: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    unk_id = token_to_id[UNK_TOKEN]
    sequences_ids = []
    for seq in sequences:
        sequence_ids = [token_to_id.get(token, unk_id) for token in seq]
        sequences_ids.append(sequence_ids)

    X = keras.utils.pad_sequences(sequences_ids, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post", value=0)

    adjusted_labels = get_adjusted_labels(labels, is_idiomatic)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(adjusted_labels)
    class_names = list(label_encoder.classes_)

    y = np.eye(len(class_names))[y_encoded]

    print(f"Data shapes: X={X.shape}, y={y.shape}")
    print(f"Classes: {class_names}")

    return X, y, class_names, adjusted_labels


def plot_training_history(history, output_path: Path):
    if history is None or not hasattr(history, "history"):
        print("No training history found to plot.")
        return

    try:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Training history plot saved to {output_path}")
    except Exception as e:
        print(f"Could not create training history plot: {e}")


def load_data(
    jsonl_path: Path,
) -> Tuple[List[List[str]], List[str], List[bool], List[int]]:
    sequences = []
    labels = []
    is_idiomatic_list = []
    sequence_lengths = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tokens = data.get("tokens", [])
            category = data.get("category", "unknown")
            idiomatic = data.get("isIdiomatic", False)

            if tokens:
                sequences.append(tokens)
                labels.append(category)
                is_idiomatic_list.append(idiomatic)
                sequence_lengths.append(len(tokens))

    print(f"Loaded {len(sequences)} samples")
    return sequences, labels, is_idiomatic_list, sequence_lengths


def get_adjusted_labels(labels: List[str], is_idiomatic: List[bool]) -> List[str]:
    adjusted_labels = []
    for i, label in enumerate(labels):
        if is_idiomatic[i]:
            adjusted_labels.append("no_issue")
        else:
            adjusted_labels.append(label)
    return adjusted_labels


def load_vocabulary(vocab_path: Path) -> Dict[str, int]:
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    print(f"Vocabulary size: {len(token_to_id)}")
    return token_to_id


def save_vocabulary(token_to_id: Dict[str, int], output_path: Path):
    with open(output_path, "w") as f:
        json.dump({str(k): v for k, v in token_to_id.items()}, f)
    print(f"Vocabulary saved to {output_path}")
