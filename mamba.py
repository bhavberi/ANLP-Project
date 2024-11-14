import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.optim as optim
from scipy.sparse import issparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# from utils.clean_data import process_data

from transformers.models.mamba.modeling_mamba import (
    MambaPreTrainedModel, 
    MambaModel,
    MambaCache, 
    MAMBA_INPUTS_DOCSTRING,
    MAMBA_START_DOCSTRING,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import List, Optional, Tuple, Union
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
)
from dataclasses import dataclass

_CHECKPOINT_FOR_DOC = "state-spaces/mamba-130m-hf"
_CONFIG_FOR_DOC = "MambaConfig"

models = {
    "mamba-tiny": "state-spaces/mamba-130m-hf",
    "mamba-mini": "state-spaces/mamba-370m-hf",
    "mamba-small": "state-spaces/mamba-790m-hf",
}

@dataclass
class MambaSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        cache_params (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    # cache_params: Optional[MambaCache] = None,
    cache_params: Optional[List[torch.FloatTensor]] = None
    # cache_params: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None

class MambaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # self.activation = ACT2FN[config.hidden_act]
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.out_proj.weight.data.normal_(mean=0.0, std=config.initializer_range)

        self.config = config

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = self.activation(x)
        # x = self.dropout(x)
        x = features
        x = self.out_proj(x)
        return x

@add_start_docstrings(
    """Mamba Model backbone with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.""",
    MAMBA_START_DOCSTRING,
)
class MambaForSequenceClassification(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.backbone = MambaModel(config)
        # self.classifier = MambaClassificationHead(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        # self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MAMBA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MambaSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MambaSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if inputs_embeds is None:
        #     inputs_embeds = self.backbone.embeddings(input_ids)

        # if self.backbone.gradient_checkpointing and self.training and use_cache:
        #     use_cache = False

        # if cache_params is None and use_cache:
        #     cache_params = MambaCache(
        #         self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
        #     )

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = mamba_outputs[0]
        logits = self.classifier(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                print(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        
        # if use_cache:
        #     cache_params.seqlen_offset += inputs_embeds.shape[1]
                
        if not return_dict:
            output = (pooled_logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaSequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )

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