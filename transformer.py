import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import issparse
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    AlbertForSequenceClassification,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
    def __init__(self, n_classes=1, model="bert-tiny", LoRA=False):
        super(TransformerClassifier, self).__init__()

        assert model in models.keys(), f"Model {model} not found in available models."

        if model.split("-")[0] == "bert":
            self.transformer = BertForSequenceClassification.from_pretrained(
                models[model], num_labels=n_classes
            )
        elif model.split("-")[0] == "roberta":
            self.transformer = RobertaForSequenceClassification.from_pretrained(
                models[model], num_labels=n_classes
            )
        elif model.split("-")[0] == "distilbert":
            self.transformer = DistilBertForSequenceClassification.from_pretrained(
                models[model], num_labels=n_classes
            )
        elif model.split("-")[0] == "albert":
            self.transformer = AlbertForSequenceClassification.from_pretrained(
                models[model], num_labels=n_classes
            )

        if LoRA:
            self._apply_lora()

    def _apply_lora(self):
        # LoRA configuration for PEFT
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Sequence classification task
            inference_mode=False,  # Training mode
            r=16,  # LoRA rank
            lora_alpha=32,  # Scaling parameter
            lora_dropout=0.1,  # Dropout for LoRA layers
        )

        # Apply LoRA to the sequence classification model
        self.transformer = get_peft_model(self.transformer, peft_config)

    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass through LoRA-adapted BERTForSequenceClassification
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs


class TransformerClassifierOld(nn.Module):
    def __init__(self, n_classes=1, dropout=0.3, hidden_size=128, model="bert-tiny"):
        super(TransformerClassifierOld, self).__init__()
        self.transformer = BertModel.from_pretrained(models[model])
        self.layer_norm = nn.LayerNorm(self.transformer.config.hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, input_ids, attention_mask):
        # Pass input through BERT
        _, pooled_output = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )

        # Apply layer normalization
        pooled_output = self.layer_norm(pooled_output)

        # Pass through the classifier
        output = self.classifier(pooled_output)

        return output


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
            loss = outputs.loss  # Use the loss computed by the model
            logits = outputs.logits  # Extract logits for predictions

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
                loss = outputs.loss  # Validation loss computed by the model
                logits = outputs.logits  # Logits for predictions

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

        print(f"Epoch {epoch+1}/{num_epochs}")
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


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

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

    return avg_loss, accuracy, f1, precision, recall


def main(num_epochs=5, model="bert-tiny"):
    # Process data
    csv_path = "edos_labelled_aggregated.csv"
    datasets, _, _ = process_data(csv_path, vectorize=False)

    device = setup()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # Train and evaluate models for each task
    for task in ["binary", "5-way", "11-way"]:
        print(f"\nTraining and evaluating {task} classification model")
        print(f"SMOTE applied: {datasets[task]['smote_applied']}")

        # Prepare datasets
        train_texts, train_labels = datasets[task]["train"]
        val_texts, val_labels = datasets[task]["val"]
        test_texts, test_labels = datasets[task]["test"]

        print("Making datasets")
        train_dataset = SentenceDataset(train_texts, train_labels, tokenizer)
        val_dataset = SentenceDataset(val_texts, val_labels, tokenizer)
        test_dataset = SentenceDataset(test_texts, test_labels, tokenizer)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # Initialize the model
        print("Creating model")
        n_classes = len(np.unique(train_labels))
        model = TransformerClassifier(n_classes=n_classes, model=model).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)

        save_path = f"best_model_{task}_{model}.pth"

        # Train the model
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
        model.load_state_dict(
            torch.load(save_path, map_location=device, weights_only=True)
        )

        # Evaluate on test set
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate_model(
            model, test_loader, device
        )

        # Print test results
        print(f"\nTest Results for {task} classification:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Macro F1 Score: {test_f1:.4f}")
        print(f"Macro Precision: {test_precision:.4f}")
        print(f"Macro Recall: {test_recall:.4f}")


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
        choices=models.keys(),
        help="Model to use for training (default: bert-tiny)",
    )
    args = parser.parse_args()

    main(args.num_epochs, args.model)
