import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import issparse
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# import logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.data.path.append("./nltk_data/")
import re

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

    # os.makedirs("models", exist_ok=True)

    return device

def process_data(csv_path):
    # Load the data
    # data = list(pd.read_csv(csv_path, nrows=4096)["text"])
    # labels = list(pd.read_csv(csv_path, nrows=4096)["label"])
    data = list(pd.read_csv(csv_path)["text"])
    labels = list(pd.read_csv(csv_path)["label"])
    # Clean and preprocess the text
    lemmatizer = WordNetLemmatizer()
    cleaned_data = []
    for index, sentence in enumerate(tqdm(data)):
        sentence = sentence.strip().lower()
        sentence = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', "", sentence)  # Remove URLs
        sentence = re.sub(
            r"\[(url|user)\]", "", sentence
        )  # Remove [URL] and [USER] tags
        tokenized_words = word_tokenize(sentence)
        tokenized_words = [
            re.sub(r"[^a-z0-9]", "", word) for word in tokenized_words if word
        ]  # Keep only alphanumeric characters
        tokenized_words = [
            lemmatizer.lemmatize(word) for word in tokenized_words
        ]

        if not tokenized_words:
            print(f"Empty sentence at index {index}: {data[index]}")
            continue
        
        sentence = " ".join(tokenized_words)
        
        cleaned_data.append(sentence)
    
    return cleaned_data, labels


class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # Convert sparse matrix to dense array if necessary
        self.texts = texts.toarray() if issparse(texts) else texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels

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
            "label": label,
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

        if LoRA:
            print("Applying LoRA to the model")
            self._apply_lora()
        elif freeze:
            print("Freezing the non-classification layers")
            for name, param in self.transformer.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

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
        self.transformer = get_peft_model(self.transformer, peft_config)  # type: ignore

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        # Forward pass through LoRA-adapted BERTForSequenceClassification
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states
        )
        return outputs
    
    def randomize_weights(self):
        for name, param in self.transformer.named_parameters():
            param.data = torch.randn_like(param.data)
    
    def probe(self, texts, labels, tokenizer, device, models=None):
        dataset = SentenceDataset(texts, labels, tokenizer)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        self.eval()
        train = False
        if models is not None:
            predictions = [[] for _ in range(13)]
            true_labels = [[] for _ in range(13)]
            train = False
        if models is None:
            models = [LogisticRegression(max_iter=1000) for _ in range(13)]
            train = True

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"]

                outputs = self(input_ids, attention_mask, output_hidden_states=True)
                # Stack the hidden states for all batches
                states = outputs.hidden_states
                for layer, state in enumerate(states):
                    if train:
                        state = state.detach().cpu().numpy().squeeze()
                        avg_state = np.mean(state, axis=1)
                        models[layer].fit(avg_state, labels)
                    else:
                        state = state.detach().cpu().numpy().squeeze()
                        avg_state = np.mean(state, axis=1)
                        predictions[layer].extend(models[layer].predict(avg_state))
                        true_labels[layer].extend(labels)
        if train:
            return models
        else:
            return predictions, true_labels

def main(
        model_type = "bert-base",
        apply_lora = False,
        freeze = False,
        csv_path = "sentiment_annotated.csv",
        ):
    device = setup()

    if model_type in models.keys():
        model_type = models[model_type]
    else:
        print(
            f"Model {model_type} not found in available models. Using {model_type} directly as model."
        )

    for task, count in zip(["binary", "5-way", "11-way"], [2, 5, 12]):
        print(f"Running {task} classification task")

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_type)

        # Load the data
        data, labels = process_data(csv_path)

        # Load the model
        model = TransformerClassifier(
            n_classes=count,
            model=model_type,
            LoRA=apply_lora,
            freeze=freeze,
            class_weights=np.ones(count)
        ).to(device)
        
        load_path = f"models/best_model_{task}_{model_type}"
        if "/" in model_type:
            load_path = f"models/best_model_{task}_{model_type.split('/')[-1]}"
        load_path += ".pth"

        # Load the model weights
        model.load_state_dict(torch.load(load_path, map_location=device))
        new_data = []
        new_labels = []
        # Reduce the data size so that both classes have the same number of samples
        count_of_labels = [0, 0]
        for dp, label in zip(data, labels):
            if label == 0 and count_of_labels[label] <= 5000:
                new_data.append(dp)
                new_labels.append(label)
                count_of_labels[label] += 1
            elif label == 1 and count_of_labels[label] <= 5000:
                new_data.append(dp)
                new_labels.append(label)
                count_of_labels[label] += 1

        # Divide the data into training and testing sets
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Predict
        logistic_regression_models = model.probe(train_data, train_labels, tokenizer, device, models=None)

        # Test all logistic regression models
        predictions, true_labels = model.probe(test_data, test_labels, tokenizer, device, models=logistic_regression_models)

        # Test all the models
        trained_accuracies = []
        for layer, (prediction, true_label) in enumerate(zip(predictions, true_labels)):
            accuracy = accuracy_score(true_label, prediction)
            print(f"Layer {layer} Accuracy: {accuracy}")
            trained_accuracies.append(accuracy)
        
        # Randomize the weights
        model.randomize_weights()
        random_logistic_regression_models = model.probe(train_data, train_labels, tokenizer, device, models=None)
        predictions, true_labels = model.probe(test_data, test_labels, tokenizer, device, models=random_logistic_regression_models)
        random_accuracies = []
        for layer, (prediction, true_label) in enumerate(zip(predictions, true_labels)):
            accuracy = accuracy_score(true_label, prediction)
            print(f"Random Layer {layer} Accuracy: {accuracy}")
            random_accuracies.append(accuracy)

        # Make graph of accuracies
        import matplotlib.pyplot as plt
        plt.plot(trained_accuracies)
        plt.plot(random_accuracies)
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.title("Accuracy of Logistic Regression Models")
        plt.legend(["Trained", "Random"])
        plt.savefig(f"probing_accuracy_{task}_{model_type}.png")
        plt.show()
        plt.clf()
        
if __name__ == "__main__":
    main()