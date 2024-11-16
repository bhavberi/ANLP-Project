import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from utils.clean_data import process_data

from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
)
from transformers.models.mamba.modeling_mamba import (
    MambaPreTrainedModel,
    MambaModel,
    MambaCache,
    MAMBA_INPUTS_DOCSTRING,
    MAMBA_START_DOCSTRING,
)

# Constants for documentation
_CHECKPOINT_FOR_DOC = "state-spaces/mamba-130m-hf"
_CONFIG_FOR_DOC = "MambaConfig"

models = {
    "mamba-tiny": "state-spaces/mamba-130m-hf",
    "mamba-mini": "state-spaces/mamba-370m-hf",
    "mamba-small": "state-spaces/mamba-790m-hf",
    "mamba-medium": "state-spaces/mamba-1.4b-hf",
    "mamba-large": "state-spaces/mamba-2.8b-hf"
}

def setup():
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs("models", exist_ok=True)
    return device

class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        # Convert numpy.str_ to regular Python strings
        self.texts = [str(text) for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = self._preprocess_texts()

    def _preprocess_texts(self):
        # Batch tokenization for better efficiency
        return self.tokenizer(
            self.texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt'
        )

    def get_class_weights(self):
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.labels),
            y=self.labels
        )
        return class_weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx].to(torch.long),  # Ensure long type for input_ids
            'attention_mask': self.encodings['attention_mask'][idx].to(torch.float32),  # Ensure float32 for attention mask
            'label': torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure long type for labels
        }

@dataclass
class MambaSequenceClassifierOutput(ModelOutput):
    """
    Output type of MambaForSequenceClassification.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification scores.
        cache_params (`List[torch.FloatTensor]`, *optional*):
            Cache parameters for the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of hidden states.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cache_params: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None

@add_start_docstrings(
    """Mamba Model for sequence classification tasks.""",
    MAMBA_START_DOCSTRING,
)
class MambaForSequenceClassification(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.backbone = MambaModel(config)
        
        # Add dropout and layer normalization
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # More complex classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        
        # Initialize weights
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MambaSequenceClassifierOutput:
        
        outputs = self.backbone(input_ids, **kwargs)
        hidden_states = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # Apply attention mask and mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states * mask
            pooled_output = hidden_states.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        # Apply dropout and layer norm
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.layer_norm(pooled_output)
        
        # Get logits through classifier
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if hasattr(self.config, 'class_weights'):
                weight = torch.tensor(self.config.class_weights, dtype=torch.float32).to(logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=weight)
            else:
                loss_fct = nn.CrossEntropyLoss()
            
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            # Print predictions for debugging (only occasionally)
            if torch.rand(1).item() < 0.01:  # 1% chance to print
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1)
                    print("\nDebug - Sample predictions:")
                    print(f"Logits shape: {logits.shape}")
                    print(f"Labels shape: {labels.shape}")
                    for i in range(min(5, len(preds))):
                        print(f"True: {labels[i].item()}, Pred: {preds[i].item()}, "
                              f"Logits: {logits[i].tolist()}, "
                              f"Probs: {probs[i].tolist()}")

        return MambaSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params if hasattr(outputs, "cache_params") else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
        )

def train_model(model, train_loader, val_loader, optimizer, device, task, num_epochs=5, save_path=None):
    best_val_f1 = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds, train_true = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['label'].to(device)
            )
            
            loss = outputs.loss
            
            # Print loss for debugging
            if not torch.isfinite(loss):
                print("Warning: Non-finite loss encountered")
                continue
                
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs.logits, dim=1)
            train_preds.extend(preds.cpu().tolist())
            train_true.extend(batch['label'].cpu().tolist())

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step(val_metrics['f1'])  # Update learning rate based on validation F1
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model with validation F1: {best_val_f1:.4f}")

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['label'].to(device)
            )
            
            total_loss += outputs.loss.item()
            _, preds = torch.max(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(batch['label'].cpu().tolist())
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy_score(all_true, all_preds),
        'f1': f1_score(all_true, all_preds, average='macro'),
        'precision': precision_score(all_true, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_true, all_preds, average='macro')
    }
    
    return metrics

def main(
    num_epochs=5,
    model_type="mamba-tiny",
    only_test=False,
    csv_path="edos_labelled_aggregated.csv",
    translated_text=False,
    save_path_suffix="",
    translated_and_normal=False,
):
    device = setup()

    assert model_type in models.keys(), f"Model {model_type} not found in available models."
    model_name = models[model_type]
    print(f"Using model: {model_name}")

    # Process data
    datasets, _, _ = process_data(
        csv_path,
        vectorize=False,
        translated_text=translated_text,
        use_normal_translated_both=translated_and_normal,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Train and evaluate models for each task
    for task in ["binary", "5-way", "11-way"]:
        print(f"\nTraining and evaluating {task} classification model")

        # Prepare datasets
        train_texts, train_labels = datasets[task]["train"]
        val_texts, val_labels = datasets[task]["val"]
        test_texts, test_labels = datasets[task]["test"]

        print("Creating datasets...")
        train_dataset = SentenceDataset(train_texts, train_labels, tokenizer, max_length=512)
        val_dataset = SentenceDataset(val_texts, val_labels, tokenizer, max_length=512)
        test_dataset = SentenceDataset(test_texts, test_labels, tokenizer, max_length=512)

        # Calculate class weights
        class_weights = train_dataset.get_class_weights()
        print("Original class weights:", class_weights)
        
        # Normalize and adjust class weights
        class_weights = class_weights / class_weights.sum()
        class_weights = np.clip(class_weights, 0.1, 10.0)
        print("Normalized class weights:", class_weights)

        # Initialize model with configuration
        print("Creating model...")
        n_classes = len(np.unique(train_labels))
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = n_classes
        config.class_weights = class_weights.astype(np.float32)
        
        model = MambaForSequenceClassification.from_pretrained(
            model_name,
            config=config
        ).to(device)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        save_path = f"models/best_model_{task}_{model_type}"
        if "/" in model_type:
            save_path = f"models/best_model_{task}_{model_type.split('/')[-1]}"
        save_path += save_path_suffix + ".pth"
        print(f"Model will be saved at: {save_path}")
        
        # Training phase
        if not only_test:
            best_val_f1 = 0
            for epoch in range(num_epochs):
                # Training
                model.train()
                train_loss = 0
                train_preds, train_true = [], []
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                    optimizer.zero_grad()
                    
                    outputs = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['label'].to(device)
                    )
                    
                    loss = outputs.loss
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, preds = torch.max(outputs.logits, dim=1)
                    train_preds.extend(preds.cpu().tolist())
                    train_true.extend(batch['label'].cpu().tolist())
        
                # Validation phase
                val_metrics = evaluate_model(model, val_loader, device)
                
                # Print metrics
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"Train Loss: {train_loss/len(train_loader):.4f}")
                print(f"Val Metrics: {val_metrics}")
                
                # Learning rate scheduling
                scheduler.step(val_metrics[2])  # Using F1 score for scheduling
                
                # Save best model
                if val_metrics[2] > best_val_f1:  # Compare F1 scores
                    best_val_f1 = val_metrics[2]
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved best model with validation F1: {best_val_f1:.4f}")

        # Load best model for testing
        model.load_state_dict(torch.load(save_path, map_location=device))
        
        # Final evaluation on test set
        test_metrics = evaluate_model(model, test_loader, device)
        
        print(f"\nFinal Test Results for {task} classification:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train and evaluate Mamba models for text classification.")
#     parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
#     parser.add_argument("--model", type=str, default="mamba-tiny", choices=list(models.keys()))
#     parser.add_argument("--test", action="store_true", help="Run only testing")
#     parser.add_argument("--csv_path", type=str, default="edos_labelled_aggregated.csv")
#     parser.add_argument("--translated_text", action="store_true")
#     parser.add_argument("--translated_and_normal", action="store_true")
#     parser.add_argument("--save_path_suffix", type=str, default="")
    
#     args = parser.parse_args()
#     print("\nArguments:", args)
    
#     main(
#         num_epochs=args.num_epochs,
#         model_type=args.model,
#         only_test=args.test,
#         csv_path=args.csv_path,
#         translated_text=args.translated_text,
#         save_path_suffix=args.save_path_suffix,
#         translated_and_normal=args.translated_and_normal,
#     )

main(
    num_epochs=1,
    model_type="mamba-tiny",
    only_test=False,
    csv_path="edos_labelled_aggregated.csv",
    translated_text=False,
    save_path_suffix="",
    translated_and_normal=False,
)
