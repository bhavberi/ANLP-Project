import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from utils.clean_data import process_data
from utils.utils import get_5label_from_11label, reverse_dict
from transformer import (
    setup,
    TransformerClassifier,
    SentenceDataset,
    train_model,
    tqdm,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, data_loader, device, initial, skip_0=False):
    model.eval()
    total_loss = 0
    all_preds, all_true = initial, []
    index = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if skip_0 and all_preds[index] == 0:
                index += 1
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            _, preds = torch.max(logits)
            # all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist()[0])
            all_preds[index] = preds.cpu().tolist()[0]
            index += 1

    # Calculate evaluation metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="macro")
    precision = precision_score(all_true, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_true, all_preds, average="macro")

    return avg_loss, accuracy, f1, precision, recall, all_preds


def main(max_iter=5, jump_1_11=False, model="bert-tiny"):
    # Process data
    csv_path = "edos_labelled_aggregated.csv"
    datasets, _, _ = process_data(csv_path, use_smote=False, vectorize=False)

    device = setup()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # Binary classification
    print("\nTraining and evaluating binary classification model")

    # Prepare datasets
    X_train_b, y_train_b = datasets["binary"]["train"]
    X_val_b, y_val_b = datasets["binary"]["val"]
    X_test_b, y_test_b = datasets["binary"]["test"]

    train_dataset_b = SentenceDataset(X_train_b, y_train_b, tokenizer)
    val_dataset_b = SentenceDataset(X_val_b, y_val_b, tokenizer)
    test_dataset_b = SentenceDataset(X_test_b, y_test_b, tokenizer)

    train_loader_b = DataLoader(train_dataset_b, batch_size=32, shuffle=True)
    val_loader_b = DataLoader(val_dataset_b, batch_size=1)
    test_loader_b = DataLoader(test_dataset_b, batch_size=1)

    # Train the binary model
    model_b = TransformerClassifier(n_classes=2, model=model).to(device)
    optimizer_b = optim.AdamW(model_b.parameters(), lr=2e-5)
    save_path = f"best_model_hier_binary_{model}.pth"
    train_model(
        model_b,
        train_loader_b,
        val_loader_b,
        optimizer_b,
        device,
        "binary",
        num_epochs=max_iter,
        save_path=save_path,
    )

    model_b.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )

    # Evaluate the binary model on the test set
    (
        test_loss_b,
        test_accuracy_b,
        test_f1_b,
        test_precision_b,
        test_recall_b,
        y_pred_b,
    ) = evaluate_model(model_b, test_loader_b, device, initial=np.zeros(len(y_test_b)))
    print("\nBinary classification model evaluation on Test Set:")
    print(f"Test Loss: {test_loss_b:.4f}")
    print(f"Test Accuracy: {test_accuracy_b:.4f}")
    print(f"Test Macro F1 Score: {test_f1_b:.4f}")
    print(f"Test Macro Precision: {test_precision_b:.4f}")
    print(f"Test Macro Recall: {test_recall_b:.4f}")

    # 5-way classification
    print("\nTraining and evaluating 5-way classification model")

    x_train_5, y_train_5 = datasets["5-way"]["train"]
    x_train_5 = x_train_5[y_train_5 != 0]
    y_train_5 = y_train_5[y_train_5 != 0]
    X_val_5, y_val_5 = datasets["5-way"]["val"]
    x_test_5, y_test_5 = datasets["5-way"]["test"]

    train_dataset_5 = SentenceDataset(x_train_5, y_train_5, tokenizer)
    val_dataset_5 = SentenceDataset(X_val_5, y_val_5, tokenizer)
    test_dataset_5 = SentenceDataset(x_test_5, y_test_5, tokenizer)

    train_loader_5 = DataLoader(train_dataset_5, batch_size=32, shuffle=True)
    val_loader_5 = DataLoader(val_dataset_5, batch_size=64)
    test_loader_5 = DataLoader(test_dataset_5, batch_size=64)

    model_5 = TransformerClassifier(n_classes=4, model=model).to(device)
    optimizer_5 = optim.AdamW(model_5.parameters(), lr=2e-5)
    save_path = f"best_model_hier_5way_{model}.pth"
    train_model(
        model_5,
        train_loader_5,
        val_loader_5,
        optimizer_5,
        device,
        "5-way",
        num_epochs=max_iter,
        save_path=save_path,
    )

    model_5.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )

    # Evaluate the 5-way model on the test set
    (
        test_loss_5,
        test_accuracy_5,
        test_f1_5,
        test_precision_5,
        test_recall_5,
        y_pred_5,
    ) = evaluate_model(
        model_5, test_loader_5, device, initial=y_pred_b.copy(), skip_0=True
    )
    print("\n5-way classification model evaluation on Test Set:")
    print(f"Test Loss: {test_loss_5:.4f}")
    print(f"Test Accuracy: {test_accuracy_5:.4f}")
    print(f"Test Macro F1 Score: {test_f1_5:.4f}")
    print(f"Test Macro Precision: {test_precision_5:.4f}")
    print(f"Test Macro Recall: {test_recall_5:.4f}")

    if jump_1_11:
        # 11-way classification
        print(f"\nTraining and evaluating {'11-way'} classification model from 1")

        x_train_11, y_train_11 = datasets["11-way"]["train"]
        x_train_11 = x_train_11[y_train_11 != 0]
        y_train_11 = y_train_11[y_train_11 != 0]
        X_val_11, y_val_11 = datasets["11-way"]["val"]
        x_test_11, y_test_11 = datasets["11-way"]["test"]

        train_dataset_11 = SentenceDataset(x_train_11, y_train_11, tokenizer)
        val_dataset_11 = SentenceDataset(X_val_11, y_val_11, tokenizer)
        test_dataset_11 = SentenceDataset(x_test_11, y_test_11, tokenizer)

        train_loader_11 = DataLoader(train_dataset_11, batch_size=32, shuffle=True)
        val_loader_11 = DataLoader(val_dataset_11, batch_size=64)
        test_loader_11 = DataLoader(test_dataset_11, batch_size=64)

        model_11 = TransformerClassifier(
            n_classes=len(np.unique(y_train_11)), model=model
        ).to(device)
        optimizer_11 = optim.AdamW(model_11.parameters(), lr=2e-5)
        save_path = f"best_model_hier_11way_{model}.pth"
        train_model(
            model_11,
            train_loader_11,
            val_loader_11,
            optimizer_11,
            device,
            "11-way",
            num_epochs=max_iter,
            save_path=save_path,
        )

        # Evaluate the 11-way model on the test set
        (
            test_loss_11,
            test_accuracy_11,
            test_f1_11,
            test_precision_11,
            test_recall_11,
            y_pred_11,
        ) = evaluate_model(
            model_11, test_loader_11, device, initial=y_pred_b.copy(), skip_0=True
        )
        print("\n11-way classification model evaluation on Test Set:")
        print(f"Test Loss: {test_loss_11:.4f}")
        print(f"Test Accuracy: {test_accuracy_11:.4f}")
        print(f"Test Macro F1 Score: {test_f1_11:.4f}")
        print(f"Test Macro Precision: {test_precision_11:.4f}")
        print(f"Test Macro Recall: {test_recall_11:.4f}")

        return
    
    # Checked till here

    # 11-way classification
    print(f"\nTraining and evaluating {'11-way'} classification model")

    x_train_11, y_train_11 = datasets["11-way"]["train"]
    X_val_11, y_val_11 = datasets["11-way"]["val"]
    x_test_11, y_test_11 = datasets["11-way"]["test"]

    # Create separate models for the four categories (1, 2, 3, 4)
    categories = [1, 2, 3, 4]
    models = {
        i: TransformerClassifier(
            n_classes=len(np.unique(y_train_11)), model="bert-tiny"
        ).to(device)
        for i in categories
    }
    optimizers = {i: optim.AdamW(models[i].parameters(), lr=2e-5) for i in categories}

    x_train_11, y_train_11 = datasets["11-way"]["train"]
    X_val_11, y_val_11 = datasets["11-way"]["val"]
    x_test_11, y_test_11 = datasets["11-way"]["test"]

    reverse_vector_mapping = reverse_dict(vector_mapping)
    y_train_11_5 = np.array(
        [get_5label_from_11label(reverse_vector_mapping[y]) for y in y_train_11]
    )

    datasets_11 = {
        i: (x_train_11[y_train_11_5 == i], y_train_11[y_train_11_5 == i])
        for i in categories
    }

    for category in categories:
        print(f"Training and evaluating model for category {category}")

        x_train_11_i, y_train_11_i = datasets_11[category]
        train_dataset_11_i = SentenceDataset(x_train_11_i, y_train_11_i, tokenizer)
        train_loader_11_i = DataLoader(train_dataset_11_i, batch_size=32, shuffle=True)

        models[category].train()
        optimizers[category].zero_grad()
        train_model(
            models[category],
            train_loader_11_i,
            val_loader_11,
            optimizers[category],
            device,
            f"11-way-cat{category}",
        )

    # Evaluate the 11-way model on the test set
    y_pred_11 = np.zeros(len(y_test_b))
    for i, y in enumerate(y_test_11):
        y_5 = y_pred_val_5[i]
        if y_5 != 0:
            y_pred_11[i] = models[y_5].predict(x_test_11[i].reshape(1, -1))
    test_loss_11, test_accuracy_11, test_f1_11, test_precision_11, test_recall_11 = (
        evaluate_model(models[y_5], test_loader_11, device)
    )

    print(f"\n{'11-way'} classification model evaluation on Test Set:")
    print(f"Test Loss: {test_loss_11:.4f}")
    print(f"Test Accuracy: {test_accuracy_11:.4f}")
    print(f"Test Macro F1 Score: {test_f1_11:.4f}")
    print(f"Test Macro Precision: {test_precision_11:.4f}")
    print(f"Test Macro Recall: {test_recall_11:.4f}")


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Train and evaluate transformer hierarchical models"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=5,
        help="The maximum number of iterations (default: 5)",
    )
    parser.add_argument(
        "--jump_1_11",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Whether to jump from 1 to 11 way classification",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert-tiny",
        help="The transformer model to use (default: bert-tiny)",
    )
    args = parser.parse_args()
    
    main(args.max_iter, args.jump_1_11, args.model)
