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
from transformers import AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mamba_classification import MambaForSequenceClassification
from utils.clean_data import process_data

models = {
    "mamba-tiny": "state-spaces/mamba-130m-hf",
    "mamba-mini": "state-spaces/mamba-370m-hf",
    "mamba-small": "state-spaces/mamba-790m-hf",
}

def setup():
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print("Using Random Seed:", random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        for text in tqdm(self.texts):
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encodings.append({
                "input_ids": encoding["input_ids"].flatten(),
            })
        return encodings

    def get_class_weights(self):
        return compute_class_weight("balanced", classes=np.unique(self.labels), y=self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        return {
            "input_ids": encoding["input_ids"],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

class MambaClassifier(nn.Module):
    def __init__(self, n_classes=1, model="mamba-tiny", class_weights=None):
        super(MambaClassifier, self).__init__()
        assert n_classes > 1, "Number of classes must be greater than 1."
        
        self.mamba = MambaForSequenceClassification.from_pretrained(
            model,
            num_labels=n_classes,
        )

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        outputs = self.mamba(
            input_ids=input_ids,
            labels=labels,
        )
        return outputs

def train_model(model, train_loader, val_loader, optimizer, device, task, num_epochs=5, save_path=None):
    best_val_f1 = 0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds, train_true = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            
            loss = outputs.loss
            logits = outputs.logits

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
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item()
                _, preds = torch.max(logits, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_true.extend(labels.cpu().tolist())

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_accuracy = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds, average="macro")
        val_accuracy = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="macro")

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

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
            labels = batch["label"].to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="macro")
    precision = precision_score(all_true, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_true, all_preds, average="macro")

    return avg_loss, accuracy, f1, precision, recall

def main(num_epochs=5, model_type="mamba-tiny", csv_path="edos_labelled_aggregated.csv"):
    device = setup()
    
    if model_type in models:
        model_name = models[model_type]
    else:
        print(f"Model {model_type} not found in available models. Using directly.")
        model_name = model_type

    print(f"Using model: {model_name}")

    # Process data
    datasets, _, _ = process_data(csv_path, vectorize=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for task in ["binary", "5-way", "11-way"]:
        print(f"\nProcessing {task} classification")

        train_texts, train_labels = datasets[task]["train"]
        val_texts, val_labels = datasets[task]["val"]
        test_texts, test_labels = datasets[task]["test"]

        train_dataset = SentenceDataset(train_texts, train_labels, tokenizer)
        val_dataset = SentenceDataset(val_texts, val_labels, tokenizer)
        test_dataset = SentenceDataset(test_texts, test_labels, tokenizer)

        class_weights = train_dataset.get_class_weights()

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        n_classes = len(np.unique(train_labels))
        model = MambaClassifier(
            n_classes=n_classes,
            model=model_name,
            class_weights=class_weights
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        save_path = f"models/mamba_{task}_{model_type}.pth"
        print(f"Model will be saved at: {save_path}")

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

        model.load_state_dict(torch.load(save_path, map_location=device))
        test_metrics = evaluate_model(model, test_loader, device)
        
        print(f"\nTest Results for {task} classification:")
        print(f"Loss: {test_metrics[0]:.4f}")
        print(f"Accuracy: {test_metrics[1]:.4f}")
        print(f"Macro F1: {test_metrics[2]:.4f}")
        print(f"Macro Precision: {test_metrics[3]:.4f}")
        print(f"Macro Recall: {test_metrics[4]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Mamba models for text classification.")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="mamba-tiny")
    parser.add_argument("--csv_path", type=str, default="edos_labelled_aggregated.csv")
    
    args = parser.parse_args()
    main(num_epochs=args.num_epochs, model_type=args.model, csv_path=args.csv_path)
