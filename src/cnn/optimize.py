import argparse
from pathlib import Path
import keras
from keras.api.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate
from sklearn.model_selection import train_test_split
from src.utils import (
    create_vocabulary,
    load_data,
    load_vocabulary,
    save_vocabulary,
    preprocess_data,
    MAX_SEQ_LENGTH,
    run_hyperparameter_optimization,
    save_optimization_results,
    plot_confusion_matrices,
    plot_roc_curves,
    calculate_classification_metrics,
)

BATCH_SIZE = 32
EPOCHS = 20
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2
CNN_FILTER_SIZES = [3, 4, 5]


def build_cnn_model_tunable(hp, vocab_size: int, num_classes: int) -> keras.Model:
    inputs = Input(shape=(MAX_SEQ_LENGTH,), name="input")

    embedding_dim = hp.Int("embedding_dim", min_value=32, max_value=300, step=32, default=128)
    cnn_num_filters = hp.Int("cnn_num_filters", min_value=32, max_value=256, step=32, default=128)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1, default=0.3)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log", default=1e-3)
    filter_sizes = CNN_FILTER_SIZES

    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="embedding")(inputs)

    conv_blocks = []
    for filter_size in filter_sizes:
        conv = Conv1D(
            filters=cnn_num_filters,
            kernel_size=filter_size,
            padding="valid",
            activation="relu",
            name=f"conv_{filter_size}",
        )(embedding)

        pool = GlobalMaxPooling1D(name=f"pool_{filter_size}")(conv)
        conv_blocks.append(pool)

    if len(filter_sizes) > 1:
        concatenated = Concatenate(name="concatenate")(conv_blocks)
    else:
        concatenated = conv_blocks[0]

    dropout = Dropout(dropout_rate, name="dropout")(concatenated)
    outputs = Dense(num_classes, activation="softmax", name="output")(dropout)
    model = keras.Model(inputs=inputs, outputs=outputs)

    print("\nModel Architecture:")
    model.summary()
    print(f"Total parameters: {model.count_params():,}")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main(args):
    output_dir = Path(args.output_dir)

    if args.vocab_file:
        token_to_id = load_vocabulary(args.vocab_file)
    else:
        token_to_id = create_vocabulary(args.input_file)
        save_vocabulary(token_to_id, output_dir / "cnn_vocab.json")

    sequences, categories, is_idiomatic = load_data(args.input_file)
    X, y, class_names, adjusted_labels = preprocess_data(sequences, categories, is_idiomatic, token_to_id)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

    best_model_from_tuner, best_hps, val_metrics, tuner, X_val, y_val = run_hyperparameter_optimization(
        build_model_fn=build_cnn_model_tunable,
        X=X_train,
        y=y_train,
        vocab_size=len(token_to_id),
        num_classes=len(class_names),
        output_dir=output_dir,
        project_name="cnn_tune",
        max_trials=args.max_trials,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        val_split=VALIDATION_SIZE,
    )

    print("\n--- Best Hyperparameters Found ---")
    print(f"Embedding Dimension: {best_hps.get('embedding_dim')}")
    print(f"CNN Filters: {best_hps.get('cnn_num_filters')}")
    print(f"Dropout Rate: {best_hps.get('dropout_rate')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")

    best_model = build_cnn_model_tunable(best_hps, len(token_to_id), len(class_names))
    best_model.set_weights(best_model_from_tuner.get_weights())

    print("\n--- Best Model Information ---")
    best_model.summary()
    print(f"Total parameters: {best_model.count_params():,}")

    save_optimization_results(
        model=best_model,
        hp=best_hps,
        tuner=tuner,
        out_dir=output_dir,
        X_val=X_val,
        y_val=y_val,
        class_names=class_names,
    )

    best_model.save(output_dir / "optimized_model.keras", include_optimizer=True)

    plot_confusion_matrices(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        out_dir=output_dir,
    )

    plot_roc_curves(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        out_dir=output_dir,
    )

    calculate_classification_metrics(
        model=best_model,
        X=X_test,
        y=y_test,
        class_names=class_names,
        out_dir=output_dir / "metrics",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for CNN model")
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
        help="Directory to save the optimized model and artifacts",
    )
    parser.add_argument(
        "-v",
        "--vocab-file",
        type=Path,
        help="Optional JSON file with vocabulary (will be created if not provided)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=20,
        help="Maximum number of hyperparameter configurations to try",
    )

    args = parser.parse_args()
    main(args)
