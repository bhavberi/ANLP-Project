import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import issparse
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils.clean_data import process_data

models = {
    "mamba-tiny": "state-spaces/mamba-130m-hf",
    "mamba-mini": "state-spaces/mamba-370m-hf",
    "mamba-small": "state-spaces/mamba-790m-hf",
    "mamba-medium": "state-spaces/mamba-1.4b-hf",
    "mamba-large": "state-spaces/mamba-2.8b-hf",
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
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)

    return device

class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.toarray() if issparse(texts) else texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.encodings = self._preprocess_texts()

    def _preprocess_texts(self):
        encodings = []
        for text in tqdm(self.texts, desc="Tokenizing"):
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

class MambaClassifier(nn.Module):
    def __init__(
        self,
        n_classes=1,
        model_name="state-spaces/mamba-130m-hf",
        LoRA=False,
        class_weights=None,
        freeze=False,
    ):
        super(MambaClassifier, self).__init__()

        assert n_classes > 1, "Number of classes must be greater than 1."
        assert not (LoRA and freeze), "LoRA and freeze cannot be applied together."

        # Load the Mamba configuration and set use_mambapy to True
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.use_mambapy = True  # Enable the optimized fast path

        # Load the Mamba base model
        self.transformer = AutoModel.from_pretrained(
            model_name, config=self.config
        )

        # Add a classification head
        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, n_classes)

        if LoRA:
            print("Applying LoRA to the model")
            self._apply_lora()
        elif freeze:
            print("Freezing the non-classification layers")
            for param in self.transformer.parameters():
                param.requires_grad = False

        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(
                self.transformer.device
            )
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _apply_lora(self):
        from peft import get_peft_model, LoraConfig, TaskType

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Sequence classification task
            inference_mode=False,  # Training mode
            r=16,  # LoRA rank
            lora_alpha=32,  # Scaling parameter
            lora_dropout=0.1,  # Dropout for LoRA layers
        )

        # Apply LoRA to the transformer model
        self.transformer = get_peft_model(self.transformer, peft_config)

    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass through the transformer model
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Use the [CLS] token representation
        sequence_output = outputs.last_hidden_state  # shape (batch_size, seq_length, hidden_size)
        pooled_output = sequence_output[:, 0, :]  # [CLS] token representation

        # Pass through dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = self.criterion(logits, labels) if labels is not None else None

        return {
            "loss": loss,
            "logits": logits,
        }

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    task,
    num_epochs=5,
    save_path=None,
):
    best_val_f1 = 0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds, train_true = [], []
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # Forward pass through the model (includes loss computation)
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]  # Use the loss computed by the model
            logits = outputs["logits"]  # Extract logits for predictions

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            train_preds.extend(preds.cpu().tolist())
            train_true.extend(labels.cpu().tolist())

        # Validation phase
        model.eval()
        val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                # Forward pass during evaluation
                outputs = model(input_ids, attention_mask, labels=labels)
                loss = outputs["loss"]
                logits = outputs["logits"]

                val_loss += loss.item()
                _, preds = torch.max(logits, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_true.extend(labels.cpu().tolist())

        # Calculate and print metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_accuracy = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds, average="macro")
        val_accuracy = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="macro")

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Macro F1: {train_f1:.4f}"
        )
        print(
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Macro F1: {val_f1:.4f}"
        )

        # Save the best model based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if save_path:
                torch.save(model.state_dict(), save_path)
            print(f"Saved best model for {task} classification.")

def evaluate_model(model, data_loader, device, return_preds=False):
    model.eval()
    total_loss = 0
    all_preds, all_true = [], []
    all_texts = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            # If you want to store texts, decode input_ids using tokenizer

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]  # Use dictionary key access
            logits = outputs["logits"]  # Use dictionary key access

            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    # Calculate evaluation metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="macro")
    precision = precision_score(all_true, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_true, all_preds, average="macro")

    if return_preds:
        return avg_loss, accuracy, f1, precision, recall, all_preds, all_true
    else:
        return avg_loss, accuracy, f1, precision, recall

def main(
    num_epochs=5,
    model_type="mamba-tiny",
    apply_lora=False,
    freeze=False,
    only_test=False,
    csv_path="edos_labelled_aggregated.csv",
    translated_text=False,
    save_path_suffix="",
    translated_and_normal=False,
):
    device = setup()

    if model_type in models.keys():
        model_name = models[model_type]
    else:
        model_name = model_type
        print(
            f"Model {model_type} not found in available models. Using {model_name} directly as model."
        )
    print(f"Using model: {model_name}")

    # Process data
    datasets, _, _ = process_data(
        csv_path,
        vectorize=False,
        translated_text=translated_text,
        use_normal_translated_both=translated_and_normal,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Train and evaluate models for each task
    for task in ["binary", "5-way", "11-way"]:
        print(f"\nTraining and evaluating {task} classification model")

        # Prepare datasets
        train_texts, train_labels = datasets[task]["train"]
        val_texts, val_labels = datasets[task]["val"]
        test_texts, test_labels = datasets[task]["test"]

        print("Making datasets")
        train_dataset = SentenceDataset(train_texts, train_labels, tokenizer)
        val_dataset = SentenceDataset(val_texts, val_labels, tokenizer)
        test_dataset = SentenceDataset(test_texts, test_labels, tokenizer)

        class_weights = train_dataset.get_class_weights()

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # Initialize the model
        print("Creating model")
        n_classes = len(np.unique(train_labels))
        model = MambaClassifier(
            n_classes=n_classes,
            model_name=model_name,
            class_weights=class_weights,
            LoRA=apply_lora,
            freeze=freeze,
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)

        save_path = f"models/best_model_{task}_{model_type}"
        if "/" in model_type:
            save_path = f"models/best_model_{task}_{model_type.split('/')[-1]}"
        save_path += save_path_suffix + ".pth"
        print(f"Model will be saved at: {save_path}")

        # Train the model
        if not only_test:
            train_model(
                model,
                train_loader,
                val_loader,
                optimizer,
                device,
                task,
                num_epochs=num_epochs,
                save_path=save_path,
            )

        # Load the best model
        model.load_state_dict(torch.load(save_path, map_location=device))

        # Evaluate on test set
        test_results = evaluate_model(model, test_loader, device, return_preds=True)
        test_loss, test_accuracy, test_f1, test_precision, test_recall, test_preds, test_true = test_results  # , test_texts = test_results
        
        # Print test results
        print(f"\nTest Results for {task} classification:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Macro F1 Score: {test_f1:.4f}")
        print(f"Macro Precision: {test_precision:.4f}")
        print(f"Macro Recall: {test_recall:.4f}")
        
        # Print predictions and true labels
        print("\nSample Predictions:")
        for idx in range(10):  # Change the range as needed
            pred_label = test_preds[idx]
            true_label = test_true[idx]
            # text = test_texts[idx]  # If you have the texts
            print(f"Prediction: {pred_label}, True Label: {true_label}")  # , Text: {text}
        
        # Clear memory
        model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate Mamba models for text classification."
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
        default="mamba-tiny",
        help=f"Model to use for training (default: mamba-tiny). Choices: {list(models.keys())}",
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