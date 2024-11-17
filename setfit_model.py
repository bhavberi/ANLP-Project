import os
import random
import argparse
import numpy as np

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from setfit import SetFitModel, Trainer
from datasets import Dataset
from utils.clean_data import process_data  # Assuming this module exists and works similarly

# Compatible sentence-transformer models
models = {
    "paraphrase-mpnet-base-v2": "sentence-transformers/paraphrase-mpnet-base-v2",
    "paraphrase-MiniLM-L3-v2": "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "paraphrase-MiniLM-L6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "all-distilroberta-v1": "sentence-transformers/all-distilroberta-v1",
    "all-roberta-large-v1": "sentence-transformers/all-roberta-large-v1",
}

def main(
    num_epochs=1,
    num_iterations=20,
    model_type="paraphrase-mpnet-base-v2",
    only_test=False,
    csv_path="edos_labelled_aggregated.csv",
    save_path_suffix="",
    translated_text=False,
    translated_and_normal=False,
):
    # Set random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print("Using Random Seed:", random_seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)

    if model_type in models:
        model_name = models[model_type]
    else:
        print(
            f"Model {model_type} not found in available models. Using {model_type} directly as model."
        )
        model_name = model_type
    print(f"Using model: {model_name}")

    # Process data
    datasets_dict, _, _ = process_data(
        csv_path,
        vectorize=False,
        translated_text=translated_text,
        use_normal_translated_both=translated_and_normal,
    )

    # Train and evaluate models for each task
    for task in ["binary", "5-way", "11-way"]:
        print(f"\nTraining and evaluating {task} classification model")

        # Prepare datasets
        train_texts, train_labels = datasets_dict[task]["train"]
        val_texts, val_labels = datasets_dict[task]["val"]
        test_texts, test_labels = datasets_dict[task]["test"]

        # Prepare datasets for SetFit
        train_data = Dataset.from_dict({"text": train_texts, "label": train_labels})
        val_data = Dataset.from_dict({"text": val_texts, "label": val_labels})
        test_data = Dataset.from_dict({"text": test_texts, "label": test_labels})

        # Initialize the SetFit model
        print("Creating SetFit model")
        model = SetFitModel.from_pretrained(model_name)

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            metric="accuracy",
            batch_size=16,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            seed=42,
            column_mapping={"text": "text", "label": "label"},
        )

        save_path = f"models/setfit_best_model_{task}_{model_type}"
        if "/" in model_type:
            save_path = f"models/setfit_best_model_{task}_{model_type.split('/')[-1]}"
        save_path += save_path_suffix

        # Train the model
        if not only_test:
            trainer.train()

            # Save the trained model
            model.save_pretrained(save_path)
            print(f"Saved model for {task} classification.")

        # Load the model
        model = SetFitModel.from_pretrained(save_path)

        # Evaluate on test set
        y_pred = model.predict(test_texts)
        accuracy = accuracy_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred, average="macro")
        precision = precision_score(test_labels, y_pred, average="macro", zero_division=0)
        recall = recall_score(test_labels, y_pred, average="macro")

        # Print test results
        print(f"\nTest Results for {task} classification:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1 Score: {f1:.4f}")
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall: {recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate SetFit models."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to train the classification head (default: 1)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=20,
        help="Number of contrastive learning iterations (default: 20)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="paraphrase-mpnet-base-v2",
        help=f"Model to use for training (default: paraphrase-mpnet-base-v2). Choices: {list(models.keys())}",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the model only for testing",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="edos_labelled_aggregated.csv",
        help="Path to the CSV file containing the data (default: edos_labelled_aggregated.csv)",
    )
    parser.add_argument(
        "--save_path_suffix",
        type=str,
        default="",
        help="Suffix to add to the model save path (default: '')",
    )
    parser.add_argument(
        "--translated_text",
        action="store_true",
        help="Use translated text for training",
    )
    parser.add_argument(
        "--translated_and_normal",
        action="store_true",
        help="Use both translated and normal text for training",
    )
    args = parser.parse_args()

    print("\n\n")
    print("Arguments:", args)

    main(
        num_epochs=args.num_epochs,
        num_iterations=args.num_iterations,
        model_type=args.model,
        only_test=args.test,
        csv_path=args.csv_path,
        save_path_suffix=args.save_path_suffix,
        translated_text=args.translated_text,
        translated_and_normal=args.translated_and_normal,
    )