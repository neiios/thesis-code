import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import keras
from keras.api.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate

from src.utils import (
    create_vocabulary,
    load_data,
    load_vocabulary,
    save_vocabulary,
    preprocess_data,
    plot_training_history,
    MAX_SEQ_LENGTH,
)

EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5
TEST_SIZE = 0.1  # 10% for test set
VALIDATION_SIZE = 0.2  # 20% of remaining data for validation (18% of total)
CNN_FILTER_SIZES = [3, 4, 5]
CNN_NUM_FILTERS = 128
CNN_DROPOUT_RATE = 0.3


# textcnn
def build_cnn_model(vocab_size: int, num_classes: int) -> keras.Model:
    inputs = Input(shape=(MAX_SEQ_LENGTH,), name="input")

    embedding = Embedding(
        input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LENGTH, name="embedding"
    )(inputs)

    conv_blocks = []
    for filter_size in CNN_FILTER_SIZES:
        conv = Conv1D(
            filters=CNN_NUM_FILTERS,
            kernel_size=filter_size,
            padding="valid",
            activation="relu",
            name=f"conv_{filter_size}",
        )(embedding)

        pool = GlobalMaxPooling1D(name=f"pool_{filter_size}")(conv)
        conv_blocks.append(pool)

    if len(CNN_FILTER_SIZES) > 1:
        concatenated = Concatenate(name="concatenate")(conv_blocks)
    else:
        concatenated = conv_blocks[0]

    dropout = Dropout(CNN_DROPOUT_RATE, name="dropout")(concatenated)
    outputs = Dense(num_classes, activation="softmax", name="output")(dropout)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
    )

    model.summary()
    return model


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.vocab_file:
        token_to_id = load_vocabulary(args.vocab_file)
    else:
        token_to_id = create_vocabulary(args.input_file)
        save_vocabulary(token_to_id, output_dir / "cnn_vocab.json")

    sequences, categories, is_idiomatic, sequence_lengths = load_data(args.input_file)

    X, y, class_names, adjusted_labels = preprocess_data(sequences, categories, is_idiomatic, token_to_id)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SIZE, random_state=42, stratify=y_temp
    )

    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0] / X.shape[0]:.1%} of data)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0] / X.shape[0]:.1%} of data)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0] / X.shape[0]:.1%} of data)")

    model = build_cnn_model(len(token_to_id), len(class_names))

    try:
        from keras.api.utils import plot_model

        plot_model(model, to_file=output_dir / "cnn_model_architecture.png", show_shapes=True, show_layer_names=True)
        print(f"Model architecture diagram saved to {output_dir / 'cnn_model_architecture.png'}")
    except ImportError:
        print("Skipping model architecture diagram (requires pydot and graphviz)")

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=0.0001)

    print("\n--- Training CNN Model ---")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    val_results = model.evaluate(X_val, y_val)
    print(f"\nValidation loss: {val_results[0]:.4f}")
    print(f"Validation accuracy: {val_results[1]:.4f}")
    if len(val_results) > 2:
        print(f"Validation precision: {val_results[2]:.4f}")
    if len(val_results) > 3:
        print(f"Validation recall: {val_results[3]:.4f}")

    test_results = model.evaluate(X_test, y_test)
    print(f"\nTest loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    if len(test_results) > 2:
        print(f"Test precision: {test_results[2]:.4f}")
    if len(test_results) > 3:
        print(f"Test recall: {test_results[3]:.4f}")

    eval_results = {
        "validation": {
            "loss": float(val_results[0]),
            "accuracy": float(val_results[1]),
            "precision": float(val_results[2]) if len(val_results) > 2 else None,
            "recall": float(val_results[3]) if len(val_results) > 3 else None,
        },
        "test": {
            "loss": float(test_results[0]),
            "accuracy": float(test_results[1]),
            "precision": float(test_results[2]) if len(test_results) > 2 else None,
            "recall": float(test_results[3]) if len(test_results) > 3 else None,
        },
    }

    with open(output_dir / "cnn_evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to {output_dir / 'cnn_evaluation_results.json'}")

    model_path = output_dir / "cnn_model.keras"
    model.save(model_path)
    print(f"CNN model saved to: {model_path}")

    with open(output_dir / "cnn_classes.json", "w") as f:
        json.dump([str(c) for c in class_names], f)
    print(f"Class names saved to {output_dir / 'cnn_classes.json'}")

    plot_training_history(history, output_dir / "cnn_training_history.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN model for code smell detection")
    parser.add_argument(
        "-i",
        "--input-file",
        type=Path,
        required=True,
        help="Input JSONL file with code tokens and labels",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the trained model and artifacts",
    )
    parser.add_argument(
        "-v",
        "--vocab-file",
        type=Path,
        help="Optional JSON file with vocabulary (will be created if not provided)",
    )

    args = parser.parse_args()
    main(args)
