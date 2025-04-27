import keras
import json
import numpy as np
from pathlib import Path
from tree_sitter_language_pack import get_parser

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_SEQ_LENGTH = 256


def load_model_and_classes(model_dir):
    model = keras.models.load_model(model_dir / "model.keras")
    with open(model_dir / "classes.json", "r") as f:
        class_names = json.load(f)
    return model, class_names


def load_vocabulary(vocab_path):
    with open(vocab_path, "r") as f:
        vocab_list = json.load(f)

    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, token in enumerate(vocab_list):
        if isinstance(token, str) and token:
            token_to_id[token] = i + 2

    return token_to_id


def tokenize_scala_code(code, parser):
    tree = parser.parse(bytes(code, "utf8"))

    def traverse(node):
        tokens = [node.type]
        for child in node.children:
            tokens.extend(traverse(child))
        return tokens

    return traverse(tree.root_node)


def classify_scala_code(code, model, token_to_id, class_names):
    parser = get_parser("scala")
    tokens = tokenize_scala_code(code, parser)

    unk_id = token_to_id[UNK_TOKEN]
    token_ids = [token_to_id.get(token, unk_id) for token in tokens]

    if len(token_ids) > MAX_SEQ_LENGTH:
        token_ids = token_ids[:MAX_SEQ_LENGTH]
    else:
        token_ids = token_ids + [0] * (MAX_SEQ_LENGTH - len(token_ids))

    input_data = np.array([token_ids], dtype=np.int32)

    predictions = model.predict(input_data, verbose=0)[0]

    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx]

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "all_predictions": {
            class_name: float(predictions[i])
            for i, class_name in enumerate(class_names)
        },
    }


if __name__ == "__main__":
    model_dir = Path(".")
    vocab_path = Path("../results/vocab.json")

    model, class_names = load_model_and_classes(model_dir)
    token_to_id = load_vocabulary(vocab_path)

    scala_code = """
    object NullCheck {
      def process(data: String): String = {
        if (data == null) {
          return "No data provided"
        }
        return data.toUpperCase()
      }
    }
    """

    scala_code = """
    object ExceptionHandler {
        def processData(data: String): String = {
            try {
                if (data == null || data.isEmpty()) {
                    throw new IllegalArgumentException("Data cannot be null or empty")
                }
                return data.toUpperCase()
            } catch {
                case e: IllegalArgumentException =>
                    return s"Error processing data: ${e.getMessage()}"
                case _: Exception =>
                    return "Unknown error occurred"
            }
        }
    }
    """

    scala_code = """
    def processData2(data: Option[String]): Try[String] = Try {
        data.filter(_.nonEmpty)
            .map(_.toUpperCase)
            .getOrElse(throw new IllegalArgumentException("Data cannot be null or empty"))
    }
    """

    result = classify_scala_code(scala_code, model, token_to_id, class_names)

    print(f"Code classified as: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nAll class probabilities:")
    for class_name, prob in result["all_predictions"].items():
        print(f"  {class_name}: {prob:.4f}")
