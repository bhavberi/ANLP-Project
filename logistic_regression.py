# Import necessary libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from clean_data import process_data
import argparse

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, task):
    # Initialize and train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='macro')

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    # Print results for each dataset
    print(f"\nResults for {task} classification:")
    print(f"Training - Accuracy: {train_accuracy:.4f}, Macro F1: {train_f1:.4f}")
    print(f"Validation - Accuracy: {val_accuracy:.4f}, Macro F1: {val_f1:.4f}")
    print(f"Test - Accuracy: {test_accuracy:.4f}, Macro F1: {test_f1:.4f}")

    return model

def main(use_smote):
    # Process data
    csv_path = 'edos_labelled_aggregated.csv'  # Update this path as needed
    datasets, category_mapping, vector_mapping = process_data(csv_path, use_smote=use_smote)

    # Train and evaluate models for each task
    models = {}
    for task in ['binary', '5-way', '11-way']:
        print(f"\nTraining and evaluating {task} classification model")
        print(f"SMOTE applied: {datasets[task]['smote_applied']}")
        
        # Prepare datasets
        X_train, y_train = datasets[task]['train']
        X_val, y_val = datasets[task]['val']
        X_test, y_test = datasets[task]['test']

        # Train and evaluate the model
        model = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, task)
        models[task] = model

    return models

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate logistic regression models with optional SMOTE sampling.')
    parser.add_argument('--use_smote', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to use SMOTE for data sampling (default: True)')
    args = parser.parse_args()
    
    # Run the main function with the specified SMOTE option
    trained_models = main(args.use_smote)
    print("\nTraining and evaluation completed for all tasks.")
