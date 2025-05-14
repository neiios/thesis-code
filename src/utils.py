import json
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Set
import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from keras.api.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_SEQ_LENGTH = 500

sns.set_theme(style="ticks", palette="colorblind")


def create_vocabulary(data_path: Path) -> Dict[str, int]:
    print(f"Creating vocabulary from: {data_path}")
    token_set: Set[str] = set()

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tokens = data.get("tokens", [])
            if tokens:
                token_set.update(tokens)

    token_to_id: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
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

    y = np.eye(len(class_names))[np.array(y_encoded)]

    print(f"Data shapes: X={X.shape}, y={y.shape}")
    print(f"Classes: {class_names}")

    return X, y, class_names, adjusted_labels


def plot_training_history(history, output_path: Path) -> None:
    if history is None or not hasattr(history, "history"):
        print("No training history found to plot.")
        return

    try:
        plt.figure(figsize=(12, 5))

        ax1 = plt.subplot(1, 2, 1)
        epochs = range(1, len(history.history["accuracy"]) + 1)
        sns.lineplot(x=epochs, y=history.history["accuracy"], label="Mokymo tikslumas")
        sns.lineplot(x=epochs, y=history.history["val_accuracy"], label="Validacijos tikslumas")
        plt.ylabel("Tikslumas")
        plt.xlabel("Epocha")
        plt.legend(loc="lower right")

        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax2 = plt.subplot(1, 2, 2)
        sns.lineplot(x=epochs, y=history.history["loss"], label="Mokymo nuostoliai")
        sns.lineplot(x=epochs, y=history.history["val_loss"], label="Validacijos nuostoliai")
        plt.ylabel("Nuostoliai")
        plt.xlabel("Epocha")
        plt.legend(loc="upper right")

        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

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


def save_vocabulary(token_to_id: Dict[str, int], output_path: Path) -> None:
    with open(output_path, "w") as f:
        json.dump({str(k): v for k, v in token_to_id.items()}, f)
    print(f"Vocabulary saved to {output_path}")


def plot_optimization_results(tuner: kt.Tuner, output_dir: Path) -> None:
    best_hps = tuner.get_best_hyperparameters(1)[0]
    top_k = 10
    trials = tuner.oracle.get_best_trials(top_k)
    plt.figure(figsize=(12, 10))
    hp_names = [name for name in best_hps.values.keys()]

    for i, hp_name in enumerate(hp_names):
        ax = plt.subplot(2, 2, i + 1) if i < 4 else plt.figure(figsize=(3, 3))
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

        if hp_name in ["use_bidirectional"]:
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


def run_hyperparameter_optimization(
    build_model_fn: Callable[[kt.HyperParameters, int, int], keras.Model],
    X: np.ndarray,
    y: np.ndarray,
    vocab_size: int,
    num_classes: int,
    output_dir: Path,
    project_name: str,
    max_trials: int,
    overwrite: bool = False,
    batch_size: int = 32,
    epochs: int = 20,
    test_size: float = 0.1,
    validation_size: float = 0.2,
) -> Tuple[keras.Model, kt.HyperParameters, List[float], kt.Tuner, np.ndarray, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tuner_dir = output_dir / "tuner"
    tuner_dir.mkdir(exist_ok=True)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=validation_size, random_state=42, stratify=y_temp
    )

    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0] / X.shape[0]:.1%} of data)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0] / X.shape[0]:.1%} of data)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0] / X.shape[0]:.1%} of data)")

    tuner = kt.BayesianOptimization(
        lambda hp: build_model_fn(hp, vocab_size, num_classes),
        objective="val_accuracy",
        max_trials=max_trials,
        num_initial_points=2,
        directory=str(tuner_dir),
        project_name=project_name,
        overwrite=overwrite,
    )

    tuner.search_space_summary()
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    print("\n--- Starting Hyperparameter Tuning with Bayesian Optimization ---")
    tuner.search(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        batch_size=batch_size,
        verbose=1,
    )

    best_hps = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    print("\n--- Evaluating Best Model ---")
    test_results = best_model.evaluate(X_test, y_test, verbose=1)

    return best_model, best_hps, test_results, tuner, X_test, y_test


def save_optimization_results(
    best_model: keras.Model,
    best_hps: kt.HyperParameters,
    test_results: List[float],
    tuner: kt.Tuner,
    output_dir: Path,
    model_type: str,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    class_names: List[str] | None = None,
) -> None:
    best_hps_dict = {name: best_hps.get(name) for name in best_hps.values}

    with open(output_dir / "best_hyperparameters.json", "w") as f:
        json.dump(best_hps_dict, f, indent=2)
    print(f"Best hyperparameters saved to {output_dir / 'best_hyperparameters.json'}")

    print("Generating hyperparameter optimization visualizations...")
    plot_optimization_results(tuner, output_dir)

    eval_results = {
        "test": {
            "loss": float(test_results[0]),
            "accuracy": float(test_results[1]),
            "precision": float(test_results[2]) if len(test_results) > 2 else None,
            "recall": float(test_results[3]) if len(test_results) > 3 else None,
        },
    }

    if X_test is not None and y_test is not None and class_names is not None:
        print("Generating confusion matrices...")
        plot_confusion_matrices(best_model, X_test, y_test, class_names, output_dir)
        print("Generating ROC curves...")
        plot_roc_curves(best_model, X_test, y_test, class_names, output_dir)

    with open(output_dir / "optimization_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to {output_dir / 'optimization_results.json'}")

    model_path = output_dir / "optimized_model.keras"
    best_model.save(model_path)
    print(f"Best model saved to: {model_path}")

    plot_model(
        best_model,
        to_file=str(output_dir / "optimized_model_architecture.png"),
        show_shapes=True,
        show_layer_names=True,
    )
    print(f"Model architecture diagram saved to {output_dir / 'optimized_model_architecture.png'}")


def plot_confusion_matrices(
    model: keras.Model, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str], output_dir: Path
) -> None:
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
    plt.xlabel("Prognozuojama klasė")
    plt.ylabel("Tikroji klasė")
    plt.tight_layout()
    plt.savefig(cm_dir / "confusion_matrix_overall.png")
    plt.close()

    for i, class_name in enumerate(class_names):
        plt.figure(figsize=(8, 6))

        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)

        cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_binary, display_labels=[f"Ne {class_name}", class_name])
        disp.plot(cmap="Blues", values_format="d", ax=plt.gca())

        plt.xlabel("Prognozuojama klasė")
        plt.ylabel("Tikroji klasė")

        plt.tight_layout()
        plt.savefig(cm_dir / f"confusion_matrix_{class_name}.png")
        plt.close()

    print(f"Klasifikavimo lentelės išsaugotos: {cm_dir}")


def plot_roc_curves(
    model: keras.Model, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str], output_dir: Path
) -> None:
    y_pred_probs = model.predict(X_test)

    roc_dir = output_dir / "roc_curves"
    roc_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))

    colors = plt.cm.get_cmap("tab10", len(class_names))

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors(i), lw=2, label=f"{class_name} (plotas po kreive = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Klaidingai teigiamų atvejų dažnis")
    plt.ylabel("Jautrumas")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_dir / "multiclass_roc_curve.png")
    plt.close()

    for i, class_name in enumerate(class_names):
        plt.figure(figsize=(8, 6))

        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color="darkblue", lw=2, label=f"ROC kreivė (plotas po kreive = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Klaidingai teigiamų atvejų dažnis")
        plt.ylabel("Jautrumas")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_dir / f"roc_curve_{class_name}.png")
        plt.close()

    print(f"ROC kreivės išsaugotos: {roc_dir}")
