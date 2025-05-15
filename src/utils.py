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

sns.set_theme(style="ticks", palette="colorblind")


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


def load_vocabulary(path: Path) -> Dict[str, int]:
    print(f"[Vocabulary] Loading from {path}")
    return json.loads(path.read_text())


def save_vocabulary(tok2id: Dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tok2id, indent=2, ensure_ascii=False))

    print(f"[Vocabulary] Saved → {path}")


def save_model_analysis(
    model: keras.Model,
    hp: kt.HyperParameters,
    metrics: List[float],
    out_dir: Path,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: List[str],
    model_type: str = "best",
) -> None:
    model_dir = out_dir / model_type
    model_dir.mkdir(parents=True, exist_ok=True)

    hp_json = json.dumps({k: hp.get(k) for k in hp.values}, indent=2, ensure_ascii=False)
    (model_dir / f"{model_type}_hyperparameters.json").write_text(hp_json)

    metrics_names = model.metrics_names
    metrics_summary = {name: float(value) for name, value in zip(metrics_names, metrics)}
    (model_dir / f"{model_type}_metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False)
    )

    evaluation_metrics = calculate_classification_metrics(model, X_val, y_val, class_names, model_dir)
    (model_dir / f"{model_type}_evaluation.json").write_text(
        json.dumps(evaluation_metrics, indent=2, ensure_ascii=False)
    )

    keras.utils.plot_model(model, to_file=str(model_dir / f"{model_type}_model_architecture.png"), show_shapes=True)
    plot_confusion_matrices(model, X_val, y_val, class_names, model_dir)
    plot_roc_curves(model, X_val, y_val, class_names, model_dir)

    print(f"[Save] {model_type.capitalize()} model analysis results saved in {model_dir}")


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

    plt.figure(figsize=(12, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, cmap="Blues", values_format="d")
    plt.xlabel("Prognozuota klasė")
    plt.ylabel("Tikroji klasė")
    plt.tight_layout()
    plt.savefig(cm_dir / "overall.png")
    plt.close()

    plt.figure(figsize=(12, 8))
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
