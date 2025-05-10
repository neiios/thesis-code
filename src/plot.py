import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from src.utils import create_vocabulary, load_data, get_adjusted_labels, save_vocabulary

sns.set_theme(style="ticks", palette="colorblind")


def plot_snippet_length_distribution(sequence_lengths: list[int], output_path: Path):
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(sequence_lengths, bins=50, kde=True)
    plt.xlabel("Žetonų skaičius")
    plt.ylabel("Fragmentų skaičius")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    mean_len = np.mean(sequence_lengths)
    median_len = np.median(sequence_lengths)
    plt.axvline(float(mean_len), color="r", linestyle="--", label=f"Vidurkis: {mean_len:.2f}")
    plt.axvline(float(median_len), color="g", linestyle=":", label=f"Mediana: {median_len:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Snippet length distribution plot saved to {output_path}")


def plot_class_distribution(adjusted_labels: list[str], output_path: Path):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(y=adjusted_labels, order=pd.Series(adjusted_labels).value_counts().index)
    plt.xlabel("Fragmentų skaičius")
    plt.ylabel("Klasė")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Class distribution plot saved to {output_path}")


def plot_idiomaticity_distribution(is_idiomatic_list: list[bool], output_path: Path):
    plt.figure(figsize=(6, 6))
    counts = pd.Series(is_idiomatic_list).value_counts()
    labels = ["Idiomatiškas" if idx else "Neidiomatiškas" for idx in counts.index]
    plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Idiomaticity distribution plot saved to {output_path}")


def main(args):
    print("Starting dataset analysis")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences, categories, is_idiomatic, sequence_lengths = load_data(args.input_file)

    adjusted_labels = get_adjusted_labels(categories, is_idiomatic)

    token_to_id = create_vocabulary(args.input_file)
    save_vocabulary(token_to_id, output_dir / "vocab.json")

    plot_snippet_length_distribution(sequence_lengths, output_dir / "snippet_length_distribution.png")
    plot_class_distribution(adjusted_labels, output_dir / "class_distribution.png")
    plot_idiomaticity_distribution(is_idiomatic, output_dir / "idiomaticity_distribution.png")

    stats = {
        "total_samples": len(sequences),
        "vocabulary_size": len(token_to_id),
        "avg_sequence_length": float(np.mean(sequence_lengths)),
        "median_sequence_length": float(np.median(sequence_lengths)),
        "max_sequence_length": int(np.max(sequence_lengths)),
        "min_sequence_length": int(np.min(sequence_lengths)),
        "idiomatic_count": sum(is_idiomatic),
        "non_idiomatic_count": len(is_idiomatic) - sum(is_idiomatic),
        "class_counts": {c: adjusted_labels.count(c) for c in set(adjusted_labels)},
    }

    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Dataset statistics saved to {output_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize code smell dataset")
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
        help="Directory to save analysis results and plots",
    )

    args = parser.parse_args()
    main(args)
