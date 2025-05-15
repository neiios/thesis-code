import argparse
from pathlib import Path
import keras
from keras.api.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate
from sklearn.model_selection import train_test_split
from src.utils import (
    create_vocabulary,
    load_data,
    save_vocabulary,
    preprocess_data,
    MAX_SEQ_LENGTH,
    run_hyperparameter_optimization,
    save_model_analysis,
)
import time

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25
CNN_FILTER_SIZES = [3, 4, 5]

# TODO: fix seed


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
    output_dir = Path(str(args.output_dir) + f"_{int(time.time())}")

    token_to_id = create_vocabulary(args.input_file)
    save_vocabulary(token_to_id, output_dir / "cnn_vocab.json")

    sequences, categories, is_idiomatic = load_data(args.input_file)
    X, y, class_names, adjusted_labels = preprocess_data(sequences, categories, is_idiomatic, token_to_id)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE, stratify=y_train, random_state=42
    )

    (
        best_model_from_tuner,
        best_hps,
        val_metrics,
        worst_model_from_tuner,
        worst_hps,
        worst_val_metrics,
        tuner,
    ) = run_hyperparameter_optimization(
        build_model_fn=build_cnn_model_tunable,
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val,
        vocab_size=len(token_to_id),
        num_classes=len(class_names),
        output_dir=output_dir,
        max_trials=args.max_trials,
        batch_size=args.batch_size,
        epochs=args.epochs,
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

    save_model_analysis(
        model=best_model,
        hp=best_hps,
        metrics=val_metrics,
        out_dir=output_dir,
        X=X_test,
        y=y_test,
        class_names=class_names,
        model_type="best",
    )

    best_model.save(output_dir / "optimized_model.keras", include_optimizer=True)

    worst_model = build_cnn_model_tunable(worst_hps, len(token_to_id), len(class_names))
    worst_model.set_weights(worst_model_from_tuner.get_weights())
    save_model_analysis(
        model=worst_model,
        hp=worst_hps,
        metrics=worst_val_metrics,
        out_dir=output_dir,
        X=X_test,
        y=y_test,
        class_names=class_names,
        model_type="worst",
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
        "--max-trials",
        type=int,
        required=True,
        help="Maximum number of hyperparameter configurations to try",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs for training the model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training the model",
    )

    args = parser.parse_args()
    main(args)
