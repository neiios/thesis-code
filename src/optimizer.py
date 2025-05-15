from pathlib import Path
import keras_tuner as kt
import keras
import numpy as np
from typing import Callable


def run_hyperparameter_optimization(
    build_model_fn: Callable[
        [kt.HyperParameters, dict[str, int], list[str]], keras.Model
    ],
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    vocabulary: dict[str, int],
    classes: list[str],
    max_trials: int,
    max_epochs: int,
    output_dir: Path,
) -> kt.Tuner:
    tuner = kt.BayesianOptimization(
        lambda hp: build_model_fn(hp, vocabulary, classes),
        objective=kt.Objective("val_f1_score", direction="max"),
        max_trials=max_trials,
        directory=str(output_dir / "tuner"),
    )

    early_stopping = keras.callbacks.EarlyStopping(
        "val_loss", patience=3, restore_best_weights=True
    )

    tuner.search(
        X,
        y,
        epochs=max_epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )

    tuner.results_summary(num_trials=max_trials)

    return tuner
