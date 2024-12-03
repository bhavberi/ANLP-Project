import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from scipy.sparse import issparse
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.utils.class_weight import compute_class_weight
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx

from utils.clean_data import process_data

models = {
    "bert-tiny": "prajjwal1/bert-tiny",
    "bert-mini": "prajjwal1/bert-mini",
    "bert-small": "prajjwal1/bert-small",
    "bert-medium": "prajjwal1/bert-medium",
    "bert-base": "bert-base-uncased",
    "bert-large": "bert-large-uncased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "distilbert-base": "distilbert-base-uncased",
    "albert-base": "albert-base-v2",
    "albert-large": "albert-large-v2",
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    "bert-base-multilingual": "google-bert/bert-base-multilingual-cased",
}


def setup():
    # Set random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print("Using Random Seed:", random_seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)

    return device


class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # Convert sparse matrix to dense array if necessary
        self.texts = texts.toarray() if issparse(texts) else texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.encodings = self._preprocess_texts()

    def _preprocess_texts(self):
        encodings = []
        for text in tqdm(self.texts):
            # Tokenize and encode the text
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            encodings.append(
                {
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten(),
                }
            )

        return encodings

    def get_class_weights(self):
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(self.labels), y=self.labels
        )
        return class_weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        label = self.labels[idx]

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long),
        }


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        n_classes=1,
        model="bert-tiny",
        LoRA=False,
        class_weights=None,
        freeze=False,
    ):
        super(TransformerClassifier, self).__init__()

        assert n_classes > 1, "Number of classes must be greater than 1."
        assert not (LoRA and freeze), "LoRA and freeze cannot be applied together."

        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model, num_labels=n_classes
        )

        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass through LoRA-adapted BERTForSequenceClassification
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        logits = outputs.logits
        loss = self.criterion(logits, labels) if labels is not None else None
        outputs.loss = loss
        return outputs

def quantize_onnx_model(model_path: str, save_path: str) -> str:
    # Perform dynamic quantization to uint8
    quantize_dynamic(
        model_input=model_path,
        model_output=save_path,
        weight_type=QuantType.QUInt8,
        per_channel=False,
        reduce_range=False,
    )
    
    # Verify the quantized model
    quantized_model = onnx.load(save_path)
    onnx.checker.check_model(quantized_model)
    
    print(f"Model successfully quantized and saved at: {save_path}")
    
    return save_path

def convert_onnx(model, save_path, device):
    # Export the model to ONNX
    dummy_input = (
        torch.zeros(1, 128, dtype=torch.long).to(device),
        torch.zeros(1, 128, dtype=torch.long).to(device),
    )
    torch.onnx.export(
        model.to(device),
        dummy_input,
        save_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size"},
        },
        export_params=True,
        do_constant_folding=True,
        opset_version=16,
    )

    # Load the model back from ONNX for verification
    import onnx

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)

    quantize_onnx_model(save_path, save_path)

    # print(f"Model saved at: {save_path}")

def main(
    num_epochs=5,
    model_type="bert-tiny",
    apply_lora=False,
    freeze=False,
    only_test=False,
    csv_path="edos_labelled_aggregated.csv",
    translated_text=False,
    save_path_suffix="",
    translated_and_normal=False,
):
    device = setup()

    # assert model_type in models.keys(), f"Model {model_type} not found in available models."
    if model_type in models.keys():
        model_type = models[model_type]
    else:
        print(
            f"Model {model_type} not found in available models. Using {model_type} directly as model."
        )
    print(f"Using model: {model_type}")

    # Process data
    datasets, _, _ = process_data(
        csv_path,
        vectorize=False,
        translated_text=translated_text,
        use_normal_translated_both=translated_and_normal,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_type, do_lower_case=True)

    # Train and evaluate models for each task
    for task in ["binary", "5-way", "11-way"]:
        print(f"\nTraining and evaluating {task} classification model")

        # Prepare datasets
        train_texts, train_labels = datasets[task]["test"]

        print("Making datasets")
        train_dataset = SentenceDataset(train_texts, train_labels, tokenizer)
        
        # Initialize the model
        print("Creating model")
        n_classes = len(np.unique(train_labels))
        model = TransformerClassifier(
            n_classes=n_classes,
            model=model_type,
            class_weights=train_dataset.get_class_weights(),
            LoRA=apply_lora,
            freeze=freeze,
        ).to(device).eval()

        save_path = f"models/best_model_{task}_{model_type}"
        if "/" in model_type:
            save_path = f"models/best_model_{task}_{model_type.split('/')[-1]}"
        save_path += save_path_suffix + ".pth"

        print(f"Model to be loaded from: {save_path}")

        # Load the best model
        model.load_state_dict(
            torch.load(save_path, map_location=device, weights_only=True)
        )

        # Convert the model to ONNX
        onnx_save_path = save_path.replace(".pth", ".onnx")
        convert_onnx(model, onnx_save_path, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate transformer models with optional SMOTE sampling."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert-tiny",
        help=f"Model to use for training (default: bert-tiny). Choices: {models.keys()}",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Apply LoRA to the model for training",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze the transformer model during training",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the model only for testing",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="edos_labelled_aggregated.csv",
        help="Path to the CSV file containing the data (default: edos_labelled_aggregated.csv)",
    )
    parser.add_argument(
        "--translated_text",
        action="store_true",
        help="Use translated text for training",
    )
    parser.add_argument(
        "--translated_and_normal",
        action="store_true",
        help="Use both translated and normal text for training",
    )
    parser.add_argument(
        "--save_path_suffix",
        type=str,
        default="",
        help="Suffix to add to the model save path (default: '')",
    )
    args = parser.parse_args()

    print("\n\n")
    print("Arguments:", args)

    main(
        num_epochs=args.num_epochs,
        model_type=args.model,
        apply_lora=args.lora,
        freeze=args.freeze,
        only_test=args.test,
        csv_path=args.csv_path,
        translated_text=args.translated_text,
        save_path_suffix=args.save_path_suffix,
        translated_and_normal=args.translated_and_normal,
    )
