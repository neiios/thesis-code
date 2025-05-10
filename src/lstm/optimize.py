import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import keras
from keras.api.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.utils import (
    create_vocabulary,
    load_data,
    load_vocabulary,
    save_vocabulary,
    preprocess_data,
    MAX_SEQ_LENGTH,
)

BATCH_SIZE = 32
EPOCHS = 20
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2

sns.set_theme(style="ticks", palette="colorblind")


def build_lstm_model_tunable(hp, vocab_size: int, num_classes: int) -> keras.Model:
    inputs = Input(shape=(MAX_SEQ_LENGTH,), name="input")

    embedding_dim = hp.Int("embedding_dim", min_value=32, max_value=300, step=32, default=128)
    lstm_units_1 = hp.Int("lstm_units_1", min_value=32, max_value=256, step=32, default=128)
    lstm_units_2 = hp.Int("lstm_units_2", min_value=32, max_value=256, step=32, default=128)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1, default=0.3)
    recurrent_dropout = hp.Float("recurrent_dropout", min_value=0.0, max_value=0.3, step=0.1, default=0.2)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log", default=1e-3)
    use_bidirectional = hp.Boolean("use_bidirectional", default=True)

    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="embedding")(inputs)

    if use_bidirectional:
        lstm1 = Bidirectional(
            LSTM(lstm_units_1, return_sequences=True, recurrent_dropout=recurrent_dropout, name="lstm_1")
        )(embedding)
    else:
        lstm1 = LSTM(lstm_units_1, return_sequences=True, recurrent_dropout=recurrent_dropout, name="lstm_1")(embedding)

    if use_bidirectional:
        lstm2 = Bidirectional(LSTM(lstm_units_2, name="lstm_2"))(lstm1)
    else:
        lstm2 = LSTM(lstm_units_2, name="lstm_2")(lstm1)

    dropout = Dropout(dropout_rate, name="dropout")(lstm2)
    outputs = Dense(num_classes, activation="softmax", name="output")(dropout)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
    )

    return model


def plot_optimization_results(tuner, output_dir: Path):
    best_hps = tuner.get_best_hyperparameters(1)[0]
    top_k = 10
    trials = tuner.oracle.get_best_trials(top_k)
    plt.figure(figsize=(12, 10))
    hp_names = [name for name in best_hps.values.keys()]

    for i, hp_name in enumerate(hp_names):
        ax = plt.subplot(3, 3, i + 1) if i < 9 else plt.figure(figsize=(6, 4))
        values = []
        scores = []

        for trial in trials:
            if hp_name not in trial.hyperparameters.values:
                continue
            value = trial.hyperparameters.values[hp_name]
            score = trial.score
            values.append(value)
            scores.append(score)

        value_score_pairs = sorted(zip(values, scores), key=lambda x: x[0])
        values, scores = zip(*value_score_pairs) if value_score_pairs else ([], [])

        if hp_name == "use_bidirectional":
            data_dict = {"category": [], "score": []}
            if False in values:
                data_dict["category"].append("False")
                data_dict["score"].append(np.mean([s for v, s in zip(values, scores) if not v]))
            if True in values:
                data_dict["category"].append("True")
                data_dict["score"].append(np.mean([s for v, s in zip(values, scores) if v]))
            ax = sns.barplot(x="category", y="score", data=pd.DataFrame(data_dict))
        else:
            ax = sns.scatterplot(x=values, y=scores, alpha=0.7, s=50)
            if hp_name in ["embedding_dim", "lstm_units_1", "lstm_units_2", "cnn_num_filters"]:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            if len(values) > 2:
                try:
                    sns.regplot(
                        x=np.array(values), y=np.array(scores), scatter=False, color="red", line_kws={"linestyle": "--"}
                    )
                except Exception as e:
                    print(f"Could not fit trend line for {hp_name}: {e}")

        plt.xlabel(hp_name)
        plt.ylabel("Validacijos tikslumo vertė")
        plt.tight_layout()

    plt.savefig(output_dir / "hyperparameter_importance.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    trial_scores = [trial.score for trial in tuner.oracle.trials.values() if trial.score is not None]
    trial_ids = range(len(trial_scores))

    ax = sns.scatterplot(x=trial_ids, y=trial_scores, alpha=0.7, s=50)
    plt.xlabel("Bandymas")
    plt.ylabel("Validacijos tikslumo vertė")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if len(trial_scores) > 2:
        try:
            window_size = min(5, len(trial_scores) // 2)
            if window_size > 0:
                df = pd.DataFrame({"trial_ids": trial_ids, "trial_scores": trial_scores})
                df["smoothed_scores"] = (
                    df["trial_scores"].rolling(window=window_size, center=True, min_periods=1).mean()
                )
                sns.lineplot(
                    x=df["trial_ids"][window_size - 1 :],
                    y=df["smoothed_scores"][window_size - 1 :],
                    color="red",
                    alpha=0.7,
                )
        except Exception as e:
            print(f"Could not create trend line for trials: {e}")

    plt.tight_layout()
    plt.savefig(output_dir / "optimization_progress.png")
    plt.close()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tuner_dir = output_dir / "tuner"
    tuner_dir.mkdir(exist_ok=True)

    if args.vocab_file:
        token_to_id = load_vocabulary(args.vocab_file)
    else:
        token_to_id = create_vocabulary(args.input_file)
        save_vocabulary(token_to_id, output_dir / "lstm_vocab.json")

    sequences, categories, is_idiomatic, sequence_lengths = load_data(args.input_file)
    X, y, class_names, adjusted_labels = preprocess_data(sequences, categories, is_idiomatic, token_to_id)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SIZE, random_state=42, stratify=y_temp
    )

    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0] / X.shape[0]:.1%} of data)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0] / X.shape[0]:.1%} of data)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0] / X.shape[0]:.1%} of data)")

    tuner = kt.BayesianOptimization(
        lambda hp: build_lstm_model_tunable(hp, vocab_size=len(token_to_id), num_classes=len(class_names)),
        objective="val_accuracy",
        max_trials=args.max_trials,
        num_initial_points=2,
        directory=str(tuner_dir),
        project_name="lstm_tune",
        overwrite=args.overwrite,
    )

    tuner.search_space_summary()
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    print("\n--- Starting Hyperparameter Tuning with Bayesian Optimization ---")
    tuner.search(
        X_train,
        y_train,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    best_hps = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    print("\n--- Best Hyperparameters Found ---")
    print(f"Embedding Dimension: {best_hps.get('embedding_dim')}")
    print(f"LSTM Units (Layer 1): {best_hps.get('lstm_units_1')}")
    print(f"LSTM Units (Layer 2): {best_hps.get('lstm_units_2')}")
    print(f"Dropout Rate: {best_hps.get('dropout_rate')}")
    print(f"Recurrent Dropout: {best_hps.get('recurrent_dropout')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    print(f"Use Bidirectional LSTM: {best_hps.get('use_bidirectional')}")

    best_hps_dict = {
        "embedding_dim": best_hps.get("embedding_dim"),
        "lstm_units_1": best_hps.get("lstm_units_1"),
        "lstm_units_2": best_hps.get("lstm_units_2"),
        "dropout_rate": best_hps.get("dropout_rate"),
        "recurrent_dropout": best_hps.get("recurrent_dropout"),
        "learning_rate": best_hps.get("learning_rate"),
        "use_bidirectional": best_hps.get("use_bidirectional"),
    }

    with open(output_dir / "best_hyperparameters.json", "w") as f:
        json.dump(best_hps_dict, f, indent=2)
    print(f"Best hyperparameters saved to {output_dir / 'best_hyperparameters.json'}")

    print("Generating hyperparameter optimization visualizations...")
    plot_optimization_results(tuner, output_dir)

    print("\n--- Evaluating Best Model ---")
    test_results = best_model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")

    eval_results = {
        "test": {
            "loss": float(test_results[0]),
            "accuracy": float(test_results[1]),
            "precision": float(test_results[2]) if len(test_results) > 2 else None,
            "recall": float(test_results[3]) if len(test_results) > 3 else None,
        },
    }

    with open(output_dir / "optimization_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to {output_dir / 'optimization_results.json'}")

    model_path = output_dir / "optimized_model.keras"
    best_model.save(model_path)
    print(f"Best model saved to: {model_path}")

    try:
        from keras.api.utils import plot_model

        plot_model(
            best_model,
            to_file=str(output_dir / "optimized_model_architecture.png"),
            show_shapes=True,
            show_layer_names=True,
        )
        print(f"Model architecture diagram saved to {output_dir / 'optimized_model_architecture.png'}")
    except ImportError:
        print("Skipping model architecture diagram (requires pydot and graphviz)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for LSTM model")
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing tuner results",
    )

    args = parser.parse_args()
    main(args)
