# Standard library imports
import re

# Third-party library imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Natural Language Processing libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Machine Learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# Transformers library
from transformers import BertTokenizer, BertModel

# Download required NLTK data
nltk.download('wordnet', quiet=True, download_dir='./nltk_data/')
nltk.data.path.append('./nltk_data/')

# Define text cleaning function
def clean_text(df):
    lemmatizer = WordNetLemmatizer()
    data = list(zip(df['text'], df['label_sexist'], df['label_category'], df['label_vector'], df['split']))
    val_data, train_data, test_data = [], [], []
    
    for counter, (sentence, label_sexist, label_category, label_vector, split) in enumerate(tqdm(data)):
        # Clean and preprocess the text
        sentence = sentence.strip().lower()
        sentence = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', '', sentence)
        sentence = re.sub(r'\[(URL|USER)\]', '', sentence)
        tokenized_words = word_tokenize(sentence)
        tokenized_words = [re.sub(r'[^a-z0-9]', '', word) for word in tokenized_words if word]
        tokenized_words = [lemmatizer.lemmatize(word) for word in tokenized_words]
        
        if not tokenized_words:
            print(f"Empty sentence at index {counter}: {df.iloc[counter]}")
            continue
        
        # Assign data to appropriate split
        if split == 'dev':
            val_data.append((tokenized_words, label_sexist, label_category, label_vector))
        elif split == 'train':
            train_data.append((tokenized_words, label_sexist, label_category, label_vector))
        elif split == 'test':
            test_data.append((tokenized_words, label_sexist, label_category, label_vector))
    
    return train_data, val_data, test_data

# Define custom lemmatizer for TfidfVectorizer
class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]

# Load and preprocess data
df = pd.read_csv('edos_labelled_aggregated.csv')
pd.set_option('display.max_colwidth', None)
train_data, val_data, test_data = clean_text(df)

# Combine all data for potential future use
all_data = train_data + val_data + test_data

# Print sample data from each split
print("Sample train data:", train_data[0][0])
print("Sample validation data:", val_data[0][0])
print("Sample test data:", test_data[0][0])
print("Data split sizes:", len(train_data), len(val_data), len(test_data))

# Prepare text and labels for model
def prepare_data(data):
    texts = np.array([' '.join(sentence) for sentence, *_ in data])
    labels = np.array([1 if label == "sexist" else 0 for _, label, *_ in data])
    return texts, labels

train_texts, train_labels = prepare_data(train_data)
val_texts, val_labels = prepare_data(val_data)
test_texts, test_labels = prepare_data(test_data)

# Initialize and fit TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=Lemmatizer(), lowercase=False)
train_texts_vectorized = vectorizer.fit_transform(train_texts)

# Apply SMOTE for handling class imbalance
smote = SMOTE(sampling_strategy='not majority')
train_texts_resampled, train_labels_resampled = smote.fit_resample(train_texts_vectorized, train_labels)


class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5):
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
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model.")

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

# Main execution
device = get_device()
print(f"Using device: {device}")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets
train_dataset = SentenceDataset(train_texts, train_labels, tokenizer)
val_dataset = SentenceDataset(val_texts, val_labels, tokenizer)
test_dataset = SentenceDataset(test_texts, test_labels, tokenizer)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Initialize the model
model = TransformerClassifier(n_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5)

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on test set
test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate_model(model, test_loader, criterion, device)

print("Test Results:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
