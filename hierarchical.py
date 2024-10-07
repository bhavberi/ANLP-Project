# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from clean_data import process_data
import argparse
import numpy as np

def main(use_smote):
    # Process data
    csv_path = 'edos_labelled_aggregated.csv'
    datasets, category_mapping, vector_mapping = process_data(csv_path, use_smote=use_smote)

    # Train and evaluate models for each task
    # ['binary', '5-way', '11-way']
    print(f"\nTraining and evaluating {'binary'} classification model")
    print(f"SMOTE applied: {datasets['binary']['smote_applied']}")
    
    # Prepare datasets
    X_train_b, y_train_b = datasets['binary']['train']
    X_test_b, y_test_b = datasets['binary']['test']

    # Train and evaluate the model
    model_b = LogisticRegression(max_iter=1000)
    model_b.fit(X_train_b, y_train_b)

    # 5-way classification
    print(f"\nTraining and evaluating {'5-way'} classification model")
    print(f"SMOTE applied: {datasets['5-way']['smote_applied']}")

    x_train_5, y_train_5 = datasets['5-way']['train']
    x_train_5 = x_train_5[y_train_5 != 0]
    y_train_5 = y_train_5[y_train_5 != 0]
    x_test_5, y_test_5 = datasets['5-way']['test']

    # Train the model
    model_5 = LogisticRegression(max_iter=1000)
    model_5.fit(x_train_5, y_train_5)

    y_pred_b = model_b.predict(X_test_b)
    test_accuracy_b = accuracy_score(y_test_b, y_pred_b)
    test_f1_b = f1_score(y_test_b, y_pred_b, average='macro')

    # initialize an empty numpy array to store the predictions for the 5-way classification
    y_pred_5 = np.zeros(len(y_pred_b))
    for i, y in enumerate(y_test_5):
        if(y != 0):
            y_pred_5[i] = model_5.predict(x_test_5[i].reshape(1, -1))
    test_accuracy_5 = accuracy_score(y_test_5, y_pred_5)
    test_f1_5 = f1_score(y_test_5, y_pred_5, average='macro')

    print(f"\n{'Binary'} classification model evaluation:")
    print(f"Test accuracy: {test_accuracy_b:.4f}")
    print(f"Test F1 score: {test_f1_b:.4f}")

    print(f"\n{'5-way'} classification model evaluation:")
    print(f"Test accuracy: {test_accuracy_5:.4f}")
    print(f"Test F1 score: {test_f1_5:.4f}")

    return

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate logistic regression models with optional SMOTE sampling.')
    parser.add_argument('--use_smote', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to use SMOTE for data sampling (default: True)')
    args = parser.parse_args()
    
    # Run the main function with the specified SMOTE option
    trained_models = main(args.use_smote)
    print("\nTraining and evaluation completed for all tasks.")
