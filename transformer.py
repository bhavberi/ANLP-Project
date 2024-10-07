import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np
from clean_data import process_data
from scipy.sparse import issparse
import argparse
import random

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print("Using Random Seed:", random_seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.toarray() if issparse(texts) else texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.texts.shape[0]

    def __getitem__(self, idx):
        text = ' '.join([str(word) for word in self.texts[idx] if word != 0])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class TransformerClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TransformerClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.fc(output)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, task, num_epochs=5):
    best_val_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds, train_true = [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            train_preds.extend(preds.cpu().tolist())
            train_true.extend(labels.cpu().tolist())

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_true.extend(labels.cpu().tolist())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_accuracy = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds, average='macro')
        val_accuracy = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='macro')

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Macro F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Macro F1: {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'best_model_{task}.pth')
            print(f"Saved best model for {task} classification.")

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average='macro')
    precision = precision_score(all_true, all_preds, average='macro')
    recall = recall_score(all_true, all_preds, average='macro')

    return avg_loss, accuracy, f1, precision, recall

def main(use_smote):
    # Process data
    csv_path = 'edos_labelled_aggregated.csv'  # Update this path as needed
    datasets, category_mapping, vector_mapping = process_data(csv_path, use_smote=use_smote)

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Train and evaluate models for each task
    for task in ['binary', '5-way', '11-way']:
        print(f"\nTraining and evaluating {task} classification model")
        print(f"SMOTE applied: {datasets[task]['smote_applied']}")
        
        # Prepare datasets
        train_texts, train_labels = datasets[task]['train']
        val_texts, val_labels = datasets[task]['val']
        test_texts, test_labels = datasets[task]['test']
        
        train_dataset = SentenceDataset(train_texts, train_labels, tokenizer)
        val_dataset = SentenceDataset(val_texts, val_labels, tokenizer)
        test_dataset = SentenceDataset(test_texts, test_labels, tokenizer)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # Initialize the model
        n_classes = len(np.unique(train_labels))
        model = TransformerClassifier(n_classes=n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)

        # Train the model
        train_model(model, train_loader, val_loader, criterion, optimizer, device, task, num_epochs=5)

        # Load the best model
        model.load_state_dict(torch.load(f'best_model_{task}.pth', map_location=device, weights_only=False))

        # Evaluate on test set
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate_model(model, test_loader, criterion, device)

        print(f"\nTest Results for {task} classification:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Macro F1 Score: {test_f1:.4f}")
        print(f"Macro Precision: {test_precision:.4f}")
        print(f"Macro Recall: {test_recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate transformer models with optional SMOTE sampling.')
    parser.add_argument('--use_smote', type=bool, default=True, help='Whether to use SMOTE for data sampling (default: True)')
    args = parser.parse_args()
    
    main(args.use_smote)
