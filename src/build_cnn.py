import keras_tuner as kt
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


def build_cnn_model(hp: kt.HyperParameters, vocabulary: dict[str, int], class_names: list[str], max_seq_length: int) -> keras.Model:
    inputs = Input(shape=(max_seq_length,), name="input")

    embedding_dim = hp.Int("embedding_dim", min_value=32, max_value=300, step=32, default=128)
    cnn_num_filters = hp.Int("cnn_num_filters", min_value=32, max_value=256, step=32, default=128)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1, default=0.3)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log", default=1e-3)
    filter_sizes = [3, 4, 5]

    embedding = Embedding(input_dim=len(vocabulary), output_dim=embedding_dim, name="embedding")(inputs)

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

    print("\nModel Architecture:")
    model.summary()
    print(f"Total parameters: {model.count_params():,}")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)  # type: ignore
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.F1Score(average="weighted", name="f1score")],
    )

    return model
