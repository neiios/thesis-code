import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_SEQ_LENGTH = 256
EMBEDDING_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    print(f"Loading vocabulary from: {vocab_path}")
    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_list = json.load(f)

        # Start indexing from 2 (after special tokens)
        for i, token in enumerate(vocab_list):
            if isinstance(token, str) and token:
                token_to_id[token] = i + 2

    print(f"Vocabulary size: {len(token_to_id)}")
    return token_to_id


def extract_label(snippet_id: str) -> str:
    try:
        return snippet_id.rsplit("_", 1)[0]
    except Exception:
        return snippet_id


def load_data(
    jsonl_path: Path, token_to_id: Dict[str, int]
) -> Tuple[List[List[int]], List[str]]:
    sequences = []
    labels = []
    skipped = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("contains_errors", False):
                    skipped += 1
                    continue

                snippet_id = data.get("id")
                tokens = data.get("tokens")

                if not snippet_id or not isinstance(tokens, list):
                    skipped += 1
                    continue

                # Convert tokens to IDs
                unk_id = token_to_id[UNK_TOKEN]
                sequence = [token_to_id.get(token, unk_id) for token in tokens]
                sequences.append(sequence)

                # Extract label
                label = extract_label(snippet_id)
                labels.append(label)

            except Exception:
                skipped += 1

    print(f"Loaded {len(sequences)} samples, skipped {skipped}")
    return sequences, labels


def preprocess_data(
    sequences: List[List[int]], labels: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X = keras.utils.pad_sequences(
        sequences, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post", value=0
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    class_names = list(label_encoder.classes_)

    y = np.eye(len(class_names))[y_encoded]

    print(f"Data shapes: X={X.shape}, y={y.shape}")
    print(f"Classes: {class_names}")

    return X, y, class_names


def build_model(vocab_size: int, num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(MAX_SEQ_LENGTH,))

    x = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True, name="embedding"
    )(inputs)

    x = keras.layers.Bidirectional(keras.layers.LSTM(LSTM_UNITS))(x)

    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()
    return model


def main(args):
    print("Starting training process (CPU only)")

    token_to_id = load_vocab(args.vocab_file)

    sequences, labels = load_data(args.input_file, token_to_id)
    X, y, class_names = preprocess_data(sequences, labels)

    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    model = build_model(len(token_to_id), len(class_names))

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    print("\n--- Training Model ---")
    model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose="1",
    )

    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"\nValidation accuracy: {accuracy:.4f}")

    if args.output_model_dir:
        output_dir = Path(args.output_model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "model.keras"
        model.save(model_path)
        print(f"Model saved to: {model_path}")

        with open(output_dir / "classes.json", "w") as f:
            json.dump([str(c) for c in class_names], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an LSTM model to classify Scala code snippets"
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=Path,
        required=True,
        help="Input JSONL file with tokens",
    )
    parser.add_argument(
        "-v",
        "--vocab-file",
        type=Path,
        required=True,
        help="Vocabulary file (JSON array of tokens)",
    )
    parser.add_argument(
        "-o",
        "--output-model-dir",
        type=Path,
        default=None,
        help="Directory to save the trained model",
    )

    args = parser.parse_args()
    main(args)
