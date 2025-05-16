import keras
import keras_tuner as kt
from keras.api.layers import (
    Input,
    Embedding,
    Dense,
    Dropout,
    LSTM,
    Bidirectional,
)


def build_lstm_model(hp: kt.HyperParameters, vocabulary: dict[str, int], class_names: list[str], max_seq_length: int) -> keras.Model:
    inputs = Input(shape=(max_seq_length,), name="input")

    embedding_dim = hp.Int("embedding_dim", min_value=32, max_value=300, step=32, default=128)
    lstm_units = hp.Int("lstm_units", min_value=32, max_value=256, step=32, default=128)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1, default=0.3)
    recurrent_dropout = hp.Float("recurrent_dropout", min_value=0.0, max_value=0.3, step=0.1, default=0.2)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log", default=1e-3)

    embedding = Embedding(input_dim=len(vocabulary), output_dim=embedding_dim, mask_zero=True, name="embedding")(inputs)
    lstm = Bidirectional(LSTM(lstm_units, return_sequences=True, recurrent_dropout=recurrent_dropout, name="lstm"))(embedding)  # type: ignore
    dropout = Dropout(dropout_rate, name="dropout")(lstm)
    outputs = Dense(len(class_names), activation="softmax", name="output")(dropout)
    model = keras.Model(inputs=inputs, outputs=outputs)

    print("\nModel Architecture:")
    model.summary()
    print(f"Total parameters: {model.count_params():,}")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)  # type: ignore
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
