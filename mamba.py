import os
import random
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
from transformers.models.mamba.modeling_mamba import MambaPreTrainedModel, MambaModel
# from utils.clean_data import process_data

models = {
    "mamba-tiny": "state-spaces/mamba-130m-hf",
    "mamba-mini": "state-spaces/mamba-370m-hf"
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
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

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
    ):
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

        return type('MambaOutput', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        })

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_true = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['label'].to(device)
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(batch['label'].cpu().tolist())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average='macro')
    precision = precision_score(all_true, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_true, all_preds, average='macro')
    
    return avg_loss, accuracy, f1, precision, recall

def main(num_epochs=5, model_type="mamba-tiny", only_test=False, csv_path="edos_labelled_aggregated.csv",
         translated_text=False, save_path_suffix="", translated_and_normal=False):
    device = setup()
    
    assert model_type in models.keys(), f"Model {model_type} not found in available models."
    model_name = models[model_type]
    print(f"Using model: {model_name}")
    
    # Process data
    datasets, _, _ = process_data(csv_path, vectorize=False, translated_text=translated_text,
                                use_normal_translated_both=translated_and_normal)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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
        
        # Calculate and normalize class weights
        class_weights = train_dataset.get_class_weights()
        class_weights = class_weights / class_weights.sum()
        class_weights = np.clip(class_weights, 0.1, 10.0)
        print("Class weights:", class_weights)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Initialize model
        print("Creating model...")
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = len(np.unique(train_labels))
        config.class_weights = class_weights.astype(np.float32)
        
        model = MambaForSequenceClassification.from_pretrained(
            model_name,
            config=config
        ).to(device)
        
        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        save_path = f"models/best_model_{task}_{model_name.split('/')[-1]}{save_path_suffix}.pth"
        print(f"Model will be saved at: {save_path}")
        
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
                val_loss, val_acc, val_f1, val_precision, val_recall = evaluate_model(model, val_loader, device)
                
                # Print metrics
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"Train Loss: {train_loss/len(train_loader):.4f}")
                print(f"Val Metrics - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
                
                # Learning rate scheduling
                scheduler.step(val_f1)
                
                # Save best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved best model with validation F1: {best_val_f1:.4f}")
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(save_path))
        test_metrics = evaluate_model(model, test_loader, device)
        
        print(f"\nTest Results for {task}:")
        metric_names = ['Loss', 'Accuracy', 'F1', 'Precision', 'Recall']
        for metric, value in zip(metric_names, test_metrics):
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main(
        num_epochs=5,
        model_type="mamba-tiny",
        only_test=False,
        csv_path="edos_labelled_aggregated.csv",
        translated_text=False,
        save_path_suffix="",
        translated_and_normal=False
    )