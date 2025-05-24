# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns


from src.reader import read_dataset, create_vocabulary, preprocess_data

sns.set_theme(style="ticks", palette="colorblind")


DATA_PATH = Path("./results/processed_full.jsonl")
OUTPUT_PATH = Path("./results/notebooks/lstm")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MAX_SEQ_LENGTH = 500
RANDOM_SEED = 2025

print(f"Loading dataset from {DATA_PATH}...")
entries = read_dataset(DATA_PATH)
print(f"Loaded {len(entries)} code snippets")

print("Creating vocabulary from the dataset...")
vocabulary = create_vocabulary(DATA_PATH)
print(f"Vocabulary size: {len(vocabulary):,}")

sequence_lengths = [len(entry.tokens) for entry in entries]
max_observed_length = max(sequence_lengths)
mean_length = sum(sequence_lengths) / len(sequence_lengths)
print(f"Maximum observed sequence length: {max_observed_length}")
print(f"Average sequence length: {mean_length:.2f}")

X, y, class_names = preprocess_data(entries, vocabulary, MAX_SEQ_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=np.argmax(y, axis=1))

print(f"Class names: {class_names}")
print("\nData split:")
print(f"- Training: {X_train.shape[0]} samples")
print(f"- Test: {X_test.shape[0]} samples")


# %%
from matplotlib.ticker import MaxNLocator


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
plt.show()
plt.savefig(OUTPUT_PATH / "snippet_length_distribution.png")
plt.close()
print(f"Snippet length distribution plot saved to {OUTPUT_PATH / 'snippet_length_distribution.png'}")

# %%
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
import seaborn as sns

y_indices = np.argmax(y, axis=1)
y_labels = [class_names[idx] for idx in y_indices]

df = pd.DataFrame({"class": y_labels})

plt.figure(figsize=(10, 6))
ax = sns.countplot(y="class", data=df, order=df["class"].value_counts().index)
plt.xlabel("Fragmentų skaičius")
plt.ylabel("Klasė")

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "class_distribution.png")
plt.show()
plt.close()
print(f"Class distribution plot saved to {OUTPUT_PATH / 'class_distribution.png'}")

# %%
import pandas as pd

is_idiomatic_list = [entry.label == "idiomatic" for entry in entries]
counts = pd.Series(is_idiomatic_list).value_counts()
labels = ["Idiomatiškas" if idx else "Neidiomatiškas" for idx in counts.index]

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "idiomaticity_distribution.png")
plt.show()
plt.close()
print(f"Idiomaticity distribution plot saved to {OUTPUT_PATH / 'idiomaticity_distribution.png'}")

# %%
import keras
from keras.api.layers import (
    Input,
    Embedding,
    Dense,
    Dropout,
    LSTM,
    Bidirectional,
)


def create_best_model() -> keras.Model:
    inputs = Input(shape=(MAX_SEQ_LENGTH,), name="input")

    embedding_dim = 256
    lstm_units = 224
    dropout_rate = 0.4
    recurrent_dropout = 0.1
    learning_rate = 0.005518397014467231

    embedding = Embedding(input_dim=len(vocabulary), output_dim=embedding_dim, mask_zero=True, name="embedding")(inputs)
    lstm = Bidirectional(LSTM(lstm_units, recurrent_dropout=recurrent_dropout, name="lstm"))(embedding)
    dropout = Dropout(dropout_rate, name="dropout")(lstm)
    outputs = Dense(len(class_names), activation="softmax", name="output")(dropout)
    model = keras.Model(inputs=inputs, outputs=outputs)

    print("\nModel Architecture:")
    model.summary()
    print(f"Total parameters: {model.count_params():,}")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.F1Score(average="weighted", name="f1score")],
    )

    return model


model = create_best_model()

# %%
from matplotlib.ticker import MaxNLocator

EPOCHS = 15

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
)
model.save(OUTPUT_PATH / "best_model.keras")


# %%
plt.figure(figsize=(10, 15))

plt.subplot(3, 1, 1)
plt.plot(history.history["accuracy"], label="Mokymo tikslumas")
plt.plot(history.history["val_accuracy"], label="Testavimo tikslumas")
plt.xlabel("Epocha")
plt.ylabel("Tikslumas")
plt.title("Modelio tikslumas")
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 3)
plt.plot(history.history["f1score"], label="Mokymo F1 statistika")
plt.plot(history.history["val_f1score"], label="Testavimo F1 statistika")
plt.xlabel("Epocha")
plt.ylabel("F1 statistika")
plt.title("Modelio F1 statistika")
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 2)
plt.plot(history.history["loss"], label="Mokymo nuostoliai")
plt.plot(history.history["val_loss"], label="Testavimo nuostoliai")
plt.xlabel("Epocha")
plt.ylabel("Nuostoliai")
plt.title("Modelio nuostoliai")
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.show()

# %%
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
plt.title("Klasifikavimo lentelė")
plt.xlabel("Prognozuota klasė")
plt.ylabel("Tikroji klasė")
plt.tight_layout()
plt.show()

print("Klasifikacijos ataskaita:\n")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

# %%
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, log_loss, roc_curve, auc
import matplotlib.pyplot as plt

tikslumas = accuracy_score(y_true, y_pred)
preciziskumas = precision_score(y_true, y_pred, average="macro", zero_division=0)
f1_statistika = f1_score(y_true, y_pred, average="macro")
svertine_kryzmine_entropija = log_loss(y_true, y_pred_probs)

results_df = pd.DataFrame(
    {
        "Metrika": [
            "Tikslumas",
            "Preciziškumas",
            "F1 statistika",
            "Svertinė kryžminė entropija",
        ],
        "Reikšmė": [tikslumas, preciziskumas, f1_statistika, svertine_kryzmine_entropija],
    }
)

results_path = OUTPUT_PATH / "metrics_table.tex"
results_df.to_latex(results_path, index=False, float_format="%.4f")
print(f"Metrikų lentelė išsaugota: {results_path}")

plt.figure(figsize=(8, 6))
for i, class_name in enumerate(class_names):
    y_true_bin = (y_true == i).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Netikro teigiamo rodiklis (FPR)")
plt.ylabel("Tikro teigiamo rodiklis (TPR)")
plt.title("AUC-ROC kreivė")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "auc_roc.png")
plt.show()
print(f"AUC-ROC kreivė išsaugota: {OUTPUT_PATH / 'auc_roc.png'}")


# %%
from IPython.display import Image, display

keras.utils.plot_model(model, to_file=str(OUTPUT_PATH / "model_architecture.png"), show_shapes=True, dpi=200)

print(f"Modelio architektūros diagrama išsaugota: {OUTPUT_PATH / 'model_architecture.png'}")
display(Image(filename=str(OUTPUT_PATH / "model_architecture.png")))
