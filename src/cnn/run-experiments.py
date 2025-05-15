import argparse
from pathlib import Path
import keras
from keras.api.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
    Concatenate,
)
from sklearn.model_selection import train_test_split
import numpy as np
import time
import keras_tuner as kt

from src.reader import read_dataset, create_vocabulary, preprocess_data, save_vocabulary
from src.optimizer import run_hyperparameter_optimization

MAX_SEQ_LENGTH = 500
RANDOM_SEED = 2025

keras.utils.set_random_seed(RANDOM_SEED)


def build_cnn_model(
    hp: kt.HyperParameters, vocabulary: dict[str, int], class_names: list[str]
) -> keras.Model:
    inputs = Input(shape=(MAX_SEQ_LENGTH,), name="input")

    embedding_dim = hp.Int(
        "embedding_dim", min_value=32, max_value=300, step=32, default=128
    )
    cnn_num_filters = hp.Int(
        "cnn_num_filters", min_value=32, max_value=256, step=32, default=128
    )
    dropout_rate = hp.Float(
        "dropout_rate", min_value=0.1, max_value=0.5, step=0.1, default=0.3
    )
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log", default=1e-3
    )
    filter_sizes = [3, 4, 5]

    embedding = Embedding(
        input_dim=len(vocabulary), output_dim=embedding_dim, name="embedding"
    )(inputs)

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

    concatenated = Concatenate(name="concatenate")(conv_blocks)
    dropout = Dropout(dropout_rate, name="dropout")(concatenated)
    outputs = Dense(len(class_names), activation="softmax", name="output")(dropout)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main(args):
    output_dir = Path(args.output_dir + f"_{int(time.time())}")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_file = Path(args.input_file)

    vocabulary = create_vocabulary(input_file)
    save_vocabulary(vocabulary, output_dir / "vocabulary.json")

    entries = read_dataset(input_file)
    X, y, class_names = preprocess_data(entries, vocabulary, MAX_SEQ_LENGTH)

    # 60/20/20 train/val/test split
    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=np.argmax(y, axis=1),
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool,
        y_train_pool,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=np.argmax(y_train_pool, axis=1),
    )

    tuner = run_hyperparameter_optimization(
        build_model_fn=build_cnn_model,
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val,
        vocabulary=vocabulary,
        classes=class_names,
        max_trials=args.max_trials,
        max_epochs=args.epochs,
        output_dir=output_dir,
    )

    models = tuner.get_best_models(num_models=args.max_trials)
    final_model: keras.Model = models[0]
    final_model.fit(x=X_train_pool, y=y_train_pool, epochs=args.epochs)

    final_model.evaluate(x=X_test, y=y_test)
    final_model.save(output_dir / "best_model.keras")
    print(f"[Model] Saved → {output_dir / 'best_model.keras'}")

    all_hps = tuner.get_best_hyperparameters(num_trials=args.max_trials)
    print(f"[Model] Best hyperparameters: {all_hps[0].values}")
    print(f"[Model] Worst hyperparameters: {all_hps[-1].values}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters for CNN model"
    )
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Input JSONL file with code tokens and labels",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory to save the optimized model and artifacts",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=20,
        help="Maximum number of trials for hyperparameter optimization",
    )

    args = parser.parse_args()
    main(args)
