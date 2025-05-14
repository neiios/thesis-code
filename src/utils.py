import json
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
import keras_tuner as kt


PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
MAX_SEQ_LENGTH = 500
sns.set_theme(style="ticks", palette="colorblind")


def create_vocabulary(data_path: Path) -> Dict[str, int]:
    token_set: Set[str] = set()

    with data_path.open(encoding="utf-8") as fh:
        for line in fh:
            token_set.update(json.loads(line).get("tokens", []))

    tok2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    tok2id.update({tok: i + 2 for i, tok in enumerate(sorted(token_set))})

    print(f"[Vocabulary] Total unique tokens: {len(tok2id):,}")
    return tok2id


def get_adjusted_labels(labels: List[str], idiomatic: List[bool]) -> List[str]:
    return ["no_issue" if is_id else lbl for lbl, is_id in zip(labels, idiomatic)]


def preprocess_data(
    sequences: List[List[str]],
    labels: List[str],
    is_idiomatic: List[bool],
    token_to_id: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    unk = token_to_id[UNK_TOKEN]
    seq_ids = [[token_to_id.get(tok, unk) for tok in seq] for seq in sequences]

    X = keras.utils.pad_sequences(
        seq_ids, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post", value=0, dtype="int32"
    )

    adjusted = get_adjusted_labels(labels, is_idiomatic)
    le = LabelEncoder().fit(adjusted)
    indices = np.asarray(le.transform(adjusted), dtype=np.int64)
    y = np.eye(len(le.classes_), dtype=np.float32)[indices]

    print(f"[Data] X{X.shape}, y{y.shape} – classes: {list(le.classes_)}")
    return X, y, list(le.classes_), adjusted


def plot_training_history(history: keras.callbacks.History, out_path: Path) -> None:
    epochs = range(1, len(history.history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.lineplot(x=epochs, y=history.history["accuracy"], label="Apmokymo tikslumas", ax=ax1)
    sns.lineplot(x=epochs, y=history.history["val_accuracy"], label="Validacijos tikslumas", ax=ax1)
    ax1.set(xlabel="Epocha", ylabel="Tikslumas")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    sns.lineplot(x=epochs, y=history.history["loss"], label="Apmokymo nuostoliai", ax=ax2)
    sns.lineplot(x=epochs, y=history.history["val_loss"], label="Validacijos nuostoliai", ax=ax2)
    ax2.set(xlabel="Epocha", ylabel="Nuostoliai")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[Visualization] Saved → {out_path}")


def load_data(jsonl_path: Path) -> Tuple[List[List[str]], List[str], List[bool]]:
    seqs, lbls, idiom = [], [], []

    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if tokens := obj.get("tokens"):
                seqs.append(tokens)
                lbls.append(obj.get("category", "unknown"))
                idiom.append(obj.get("isIdiomatic", False))

    print(f"[Load] Samples read: {len(seqs):,}")
    return seqs, lbls, idiom


def load_vocabulary(path: Path) -> Dict[str, int]:
    print(f"[Vocabulary] Loading from {path}")
    return json.loads(path.read_text())


def save_vocabulary(tok2id: Dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tok2id, indent=2, ensure_ascii=False))

    print(f"[Vocabulary] Saved → {path}")


def run_hyperparameter_optimization(
    build_model_fn: Callable[[kt.HyperParameters, int, int], keras.Model],
    X: np.ndarray,
    y: np.ndarray,
    vocab_size: int,
    num_classes: int,
    output_dir: Path,
    project_name: str = "bayes_lstm",
    max_trials: int = 30,
    batch_size: int = 32,
    epochs: int = 20,
    val_split: float = 0.2,
) -> Tuple[keras.Model, kt.HyperParameters, List[float], kt.Tuner, np.ndarray, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)

    tuner = kt.BayesianOptimization(
        lambda hp: build_model_fn(hp, vocab_size, num_classes),
        objective="val_accuracy",
        max_trials=max_trials,
        directory=str(output_dir / "tuner"),
        project_name=project_name,
        overwrite=True,
    )

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_split, stratify=y)
    early = keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)

    tuner.search(
        X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early], verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model: keras.Model = tuner.get_best_models(1)[0]
    val_metrics = best_model.evaluate(X_val, y_val)

    return best_model, best_hp, val_metrics, tuner, X_val, y_val


def save_optimization_results(
    model: keras.Model,
    hp: kt.HyperParameters,
    tuner: kt.Tuner,
    out_dir: Path,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    hp_json = json.dumps({k: hp.get(k) for k in hp.values}, indent=2, ensure_ascii=False)
    (out_dir / "best_hyperparameters.json").write_text(hp_json)

    evaluation_metrics = calculate_classification_metrics(model, X_val, y_val, class_names, out_dir)
    (out_dir / "evaluation.json").write_text(json.dumps(evaluation_metrics, indent=2, ensure_ascii=False))

    keras.utils.plot_model(model, to_file=str(out_dir / "model_architecture.png"), show_shapes=True)

    print(f"[Save] Optimisation results saved in {out_dir}")


def plot_confusion_matrices(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    out_dir: Path,
) -> None:
    y_prob = model.predict(X_test)
    y_pred, y_true = np.argmax(y_prob, 1), np.argmax(y_test, 1)

    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, cmap="Blues", values_format="d")
    plt.xlabel("Prognozuota klasė")
    plt.ylabel("Tikroji klasė")
    plt.tight_layout()
    plt.savefig(cm_dir / "overall.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap="Blues", values_format=".2f", normalize="true"
    )
    plt.xlabel("Prognozuota klasė")
    plt.ylabel("Tikroji klasė")
    plt.tight_layout()
    plt.savefig(cm_dir / "overall_norm.png")
    plt.close()

    for i, cname in enumerate(class_names):
        plt.figure(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            (y_true == i).astype(int),
            (y_pred == i).astype(int),
            display_labels=[f"Ne {cname}", cname],
            cmap="Blues",
            values_format="d",
        )
        plt.xlabel("Prognozuota klasė")
        plt.ylabel("Tikroji klasė")
        plt.tight_layout()
        plt.savefig(cm_dir / f"{cname}.png")
        plt.close()

    print(f"[Visualization] Confusion matrices saved → {cm_dir}")


def plot_roc_curves(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    out_dir: Path,
) -> None:
    y_prob = model.predict(X_test)

    roc_dir = out_dir / "roc_curves"
    pr_dir = out_dir / "precision_recall_curves"
    roc_dir.mkdir(exist_ok=True)
    pr_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab10", len(class_names))

    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, lw=2, label=f"{cname} (AUC = {auc(fpr, tpr):.2f})", color=cmap(i))

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlabel("Klaidingai teigiamų dažnis")
    plt.ylabel("Jautrumas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_dir / "multiclass.png")
    plt.close()

    plt.figure(figsize=(10, 8))

    for i, cname in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_test[:, i], y_prob[:, i])
        ap = average_precision_score(y_test[:, i], y_prob[:, i])
        plt.plot(rec, prec, lw=2, label=f"{cname} (AP = {ap:.2f})", color=cmap(i))

    plt.xlabel("Jautrumas")
    plt.ylabel("Preciziškumas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_dir / "multiclass.png")
    plt.close()

    print(f"[Visualization] ROC curves → {roc_dir} | PR curves → {pr_dir}")


def calculate_classification_metrics(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    out_dir: Path,
) -> Dict:
    y_prob = model.predict(X)
    y_pred, y_true = np.argmax(y_prob, 1), np.argmax(y, 1)

    metrics: Dict = {
        "bendros": {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1": f1_score(y_true, y_pred, average="weighted"),
        },
        "pagal_klase": {},
    }

    for i, cname in enumerate(class_names):
        bin_true = (y_true == i).astype(int)
        bin_pred = (y_pred == i).astype(int)

        metrics["pagal_klase"][cname] = {
            "precision": precision_score(bin_true, bin_pred, zero_division=0),
            "recall": recall_score(bin_true, bin_pred, zero_division=0),
            "f1": f1_score(bin_true, bin_pred, zero_division=0),
            "roc_auc": roc_auc_score(bin_true, y_prob[:, i]),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "detailed_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out_dir / "classification_report.txt").write_text(
        str(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    )

    print(f"[Metrics] Saved metrics to {out_dir}")
    return metrics
