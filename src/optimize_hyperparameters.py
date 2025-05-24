import argparse
from pathlib import Path
import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import time
from typing import Callable

from src.reader import read_dataset, create_vocabulary, preprocess_data, save_vocabulary
from src.build_cnn import build_cnn_model
from src.build_lstm import build_lstm_model

MAX_SEQ_LENGTH = 500
RANDOM_SEED = 2025

keras.utils.set_random_seed(RANDOM_SEED)


def run_hyperparameter_optimization(
    build_model_fn: Callable[[kt.HyperParameters, dict[str, int], list[str], int], keras.Model],
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    vocabulary: dict[str, int],
    classes: list[str],
    max_seq_length: int,
    max_trials: int,
    max_epochs: int,
    output_dir: Path,
) -> kt.Tuner:
    tuner = kt.BayesianOptimization(
        lambda hp: build_model_fn(hp, vocabulary, classes, max_seq_length),
        objective=kt.Objective("val_f1score", direction="max"),
        max_trials=max_trials,
        directory=str(output_dir / "tuner"),
    )

    early_stopping = keras.callbacks.EarlyStopping("val_loss", patience=3, restore_best_weights=True)

    tuner.search(
        X,
        y,
        epochs=max_epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )

    return tuner


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

    print(f"Class names: {class_names}")
    print("\nData split:")
    print(f"- Training: {X_train.shape[0]} samples")
    print(f"- Validation: {X_val.shape[0]} samples")
    print(f"- Test: {X_test.shape[0]} samples")

    tuner = run_hyperparameter_optimization(
        build_model_fn=build_lstm_model if args.lstm else build_cnn_model,
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val,
        vocabulary=vocabulary,
        classes=class_names,
        max_seq_length=MAX_SEQ_LENGTH,
        max_trials=args.max_trials,
        max_epochs=args.epochs,
        output_dir=output_dir,
    )

    original_stdout = sys.stdout
    results_file = output_dir / "hyperparameter_results.txt"
    with open(results_file, "w") as f:
        sys.stdout = f
        tuner.results_summary(num_trials=args.max_trials)

        all_hps = tuner.get_best_hyperparameters(num_trials=args.max_trials)
        f.write("\n\n")
        f.write("Best hyperparameters:\n")
        f.write(f"{all_hps[0].values}\n")

        f.write("\nWorst hyperparameters:\n")
        f.write(f"{all_hps[-1].values}\n")
    sys.stdout = original_stdout

    print(f"[Model] Best hyperparameters: {all_hps[0].values}")
    print(f"[Model] Worst hyperparameters: {all_hps[-1].values}")
    print(f"[Model] Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for CNN model")
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
    parser.add_argument("--lstm", type=bool, default=False)

    args = parser.parse_args()
    main(args)
