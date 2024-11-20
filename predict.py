import random
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from scipy.sparse import issparse
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
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
    data = list(pd.read_csv(csv_path, nrows=4096)["text"])
    # data = list(pd.read_csv(csv_path)["text"])
    # Clean and preprocess the text
    lemmatizer = WordNetLemmatizer()
    cleaned_data = []
    for index, sentence in enumerate(tqdm(data)):
        sentence = sentence.strip().lower()
        sentence = re.sub(
            r'https?://[^\s<>"]+|www\.[^\s<>"]+', "", sentence
        )  # Remove URLs
        sentence = re.sub(
            r"\[(url|user)\]", "", sentence
        )  # Remove [URL] and [USER] tags
        tokenized_words = word_tokenize(sentence)
        tokenized_words = [
            re.sub(r"[^a-z0-9]", "", word) for word in tokenized_words if word
        ]  # Keep only alphanumeric characters
        tokenized_words = [lemmatizer.lemmatize(word) for word in tokenized_words]

        if not tokenized_words:
            print(f"Empty sentence at index {index}: {data[index]}")
            continue

        sentence = " ".join(tokenized_words)

        cleaned_data.append(sentence)

    return cleaned_data


class SentenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        # Convert sparse matrix to dense array if necessary
        self.texts = texts.toarray() if issparse(texts) else texts
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

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
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

    def forward(self, input_ids, attention_mask):
        # Forward pass through LoRA-adapted BERTForSequenceClassification
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def predict(self, text, tokenizer, device):
        self.eval()
        text_encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_encoding["input_ids"].to(device)
        attention_mask = text_encoding["attention_mask"].to(device)
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1)
        return prediction.cpu().numpy()[0]


def main(model_type="bert-base", apply_lora=False, freeze=False):
    device = setup()

    binary_mapping = {"Not sexist": 0, "Sexist": 1}
    category_mapping = {
        "Not sexist": 0,
        "3. Animosity": 1,
        "2. Derogation": 2,
        "4. Prejudiced discussions": 3,
        "1. Threats, plans to harm and incitement": 4,
    }
    vector_mapping = {
        "Not sexist": 0,
        "3.3 Backhanded gendered compliments": 1,
        "2.3 Dehumanising attacks & overt sexual objectification": 2,
        "2.1 Descriptive attacks": 3,
        "2.2 Aggressive and emotive attacks": 4,
        "4.2 Supporting systemic discrimination against women as a group": 5,
        "1.2 Incitement and encouragement of harm": 6,
        "4.1 Supporting mistreatment of individual women": 7,
        "3.2 Immutable gender differences and gender stereotypes": 8,
        "3.1 Casual use of gendered slurs, profanities, and insults": 9,
        "1.1 Threats of harm": 10,
        "3.4 Condescending explanations or unwelcome advice": 11,
    }
    if model_type in models.keys():
        model_type = models[model_type]
    else:
        print(
            f"Model {model_type} not found in available models. Using {model_type} directly as model."
        )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    # Load the model
    binary_model = TransformerClassifier(
        n_classes=2,
        model=model_type,
        LoRA=apply_lora,
        freeze=freeze,
        class_weights=np.ones(2),
    ).to(device)
    five_way_model = TransformerClassifier(
        n_classes=5,
        model=model_type,
        LoRA=apply_lora,
        freeze=freeze,
        class_weights=np.ones(5),
    ).to(device)
    twelve_way_model = TransformerClassifier(
        n_classes=12,
        model=model_type,
        LoRA=apply_lora,
        freeze=freeze,
        class_weights=np.ones(12),
    ).to(device)

    load_path = f"models/best_model_{"binary"}_{model_type}"
    if "/" in model_type:
        load_path = f"models/best_model_{"binary"}_{model_type.split('/')[-1]}"
    load_path += ".pth"
    binary_model.load_state_dict(torch.load(load_path, map_location=device))

    load_path = f"models/best_model_{"5-way"}_{model_type}"
    if "/" in model_type:
        load_path = f"models/best_model_{"5-way"}_{model_type.split('/')[-1]}"
    load_path += ".pth"
    five_way_model.load_state_dict(torch.load(load_path, map_location=device))

    load_path = f"models/best_model_{"11-way"}_{model_type}"
    if "/" in model_type:
        load_path = f"models/best_model_{"11-way"}_{model_type.split('/')[-1]}"
    load_path += ".pth"
    twelve_way_model.load_state_dict(torch.load(load_path, map_location=device))

    while True:
        # Get input text
        data = input("\n> ")
        if data == "exit":
            break

        # Predict Binary
        prediction = binary_model.predict(data, tokenizer, device)
        for key, value in binary_mapping.items():
            if value == prediction:
                prediction = key
                break
        print("Binary Model output:", prediction)

        # Predict 5-way
        prediction = five_way_model.predict(data, tokenizer, device)
        for key, value in category_mapping.items():
            if value == prediction:
                prediction = key
                break
        print("5-way Model output:", prediction)

        # Predict 12-way
        prediction = twelve_way_model.predict(data, tokenizer, device)
        for key, value in vector_mapping.items():
            if value == prediction:
                prediction = key
                break
        print("12-way Model output:", prediction)


if __name__ == "__main__":
    # Add arguments to the parser
    parser = argparse.ArgumentParser(description="Predict sentiment from text")
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert-base",
        help="Model type to use for prediction",
    )
    parser.add_argument(
        "--apply_lora",
        action="store_true",
        help="Apply LoRA to the model",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze the non-classification layers",
    )
    args = parser.parse_args()

    main(args.model_type, args.apply_lora, args.freeze)
