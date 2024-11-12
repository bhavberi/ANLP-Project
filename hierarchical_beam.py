# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from utils.clean_data import process_data, apply_smote
from utils.utils import get_5label_from_11label, reverse_dict
import argparse
import numpy as np


def main(use_smote, max_iter=1000, jump_1_11=False):
    # Process data
    csv_path = "edos_labelled_aggregated.csv"
    datasets, category_mapping, vector_mapping = process_data(
        csv_path, use_smote=use_smote
    )
    print("Categories mapping:")
    for key, value in category_mapping.items():
        print(key, value)
    print("Vector mapping:")
    for key, value in vector_mapping.items():
        print(key, value)
    # Binary classification
    print("\nTraining and evaluating binary classification model")
    print(f"SMOTE applied: {datasets['binary']['smote_applied']}")

    # Prepare datasets
    X_train_b, y_train_b = datasets["binary"]["train"]
    X_val_b, y_val_b = datasets["binary"]["val"]
    X_test_b, y_test_b = datasets["binary"]["test"]

    X_train_b_resampled, y_train_b_resampled = apply_smote(X_train_b, y_train_b)  # type: ignore

    # Train the binary model
    model_b = LogisticRegression(max_iter=max_iter)
    model_b.fit(X_train_b_resampled, y_train_b_resampled)

    # Evaluate the binary model in Validation set
    y_pred_val_b = model_b.predict(X_val_b)
    val_accuracy_b = accuracy_score(y_val_b, y_pred_val_b)
    val_f1_b = f1_score(y_val_b, y_pred_val_b, average="macro")

    # Evaluate the binary model in Test set
    y_pred_b = model_b.predict(X_test_b)
    test_accuracy_b = accuracy_score(y_test_b, y_pred_b)
    test_f1_b = f1_score(y_test_b, y_pred_b, average="macro")

    # Print overall evaluation for binary model
    print("\nBinary classification model evaluation on Validation Set:")
    print(f"Validation accuracy: {val_accuracy_b:.4f}")
    print(f"Validation F1 score: {val_f1_b:.4f}")
    print("\nBinary classification model evaluation on Test Set:")
    print(f"Test accuracy: {test_accuracy_b:.4f}")
    print(f"Test F1 score: {test_f1_b:.4f}")

    # 5-way classification
    print("\nTraining and evaluating 5-way classification model")
    print(f"SMOTE applied: {datasets['5-way']['smote_applied']}")

    x_train_5, y_train_5 = datasets["5-way"]["train"]
    X_val_5, y_val_5 = datasets["5-way"]["val"]
    x_test_5, y_test_5 = datasets["5-way"]["test"]

    x_train_5_resampled, y_train_5_resampled = apply_smote(x_train_5, y_train_5)  # type: ignore

    # Train the model
    model_5 = LogisticRegression(max_iter=max_iter)
    model_5.fit(x_train_5_resampled, y_train_5_resampled)

    y_pred_5 = y_pred_b.copy()
    for i in range(len(y_test_5)):
        b_probs = list(model_b.predict_proba(X_test_b[i])[0])
        b_probs = b_probs + [b_probs[1]] * 3
        y_probs = model_5.predict_proba(x_test_5[i])[0]
        final_probs = [a*b for a, b in zip(b_probs, y_probs)]
        y_pred_5[i] = max(range(len(final_probs)), key=final_probs.__getitem__)
    test_accuracy_5 = accuracy_score(y_test_5, y_pred_5)
    test_f1_5 = f1_score(y_test_5, y_pred_5, average="macro")

    y_pred_val_5 = y_pred_val_b.copy()
    for i in range(len(y_val_5)):
        b_probs = list(model_b.predict_proba(X_val_b[i])[0])
        b_probs = b_probs + [b_probs[1]] * 3
        y_probs = model_5.predict_proba(X_val_5[i])[0]
        final_probs = [a*b for a, b in zip(b_probs, y_probs)]
        y_pred_val_5[i] = max(range(len(final_probs)), key=final_probs.__getitem__)
    val_accuracy_5 = accuracy_score(y_val_5, y_pred_val_5)
    val_f1_5 = f1_score(y_val_5, y_pred_val_5, average="macro")

    # Print overall evaluation for 5-way model
    print("\n5-way classification model evaluation on Validation Set:")
    print(f"Validation accuracy: {val_accuracy_5:.4f}")
    print(f"Validation F1 score: {val_f1_5:.4f}")
    print("\n5-way classification model evaluation on Test Set:")
    print(f"Test accuracy: {test_accuracy_5:.4f}")
    print(f"Test F1 score: {test_f1_5:.4f}")

    x_train_11, y_train_11 = datasets["11-way"]["train"]
    X_val_11, y_val_11 = datasets["11-way"]["val"]
    x_test_11, y_test_11 = datasets["11-way"]["test"]

    x_train_11_resampled, y_train_11_resampled = apply_smote(x_train_11, y_train_11)  # type: ignore

    # Train the model
    model_11 = LogisticRegression(max_iter=max_iter)
    model_11.fit(x_train_11_resampled, y_train_11_resampled)

    y_pred_11 = y_pred_b.copy()
    for i in range(len(y_test_11)):
        b_probs = list(model_b.predict_proba(X_test_b[i])[0])
        b_probs = b_probs + [b_probs[1]] * 10
        y_probs_temp = model_5.predict_proba(x_test_5[i])[0]
        y_probs = [0] * 12
        y_probs[0] = y_probs_temp[0]
        y_probs[1] = y_probs_temp[1]
        y_probs[2] = y_probs_temp[2]
        y_probs[3] = y_probs_temp[2]
        y_probs[4] = y_probs_temp[2]
        y_probs[5] = y_probs_temp[3]
        y_probs[6] = y_probs_temp[4]
        y_probs[7] = y_probs_temp[3]
        y_probs[8] = y_probs_temp[1]
        y_probs[9] = y_probs_temp[1]
        y_probs[10] = y_probs_temp[4]
        y_probs[11] = y_probs_temp[1]
        eleven_probs = model_11.predict_proba(x_test_11[i])[0]
        final_probs = [a*b*c for a, b, c in zip(b_probs, y_probs, eleven_probs)]
        y_pred_11[i] = max(range(len(final_probs)), key=final_probs.__getitem__)
    test_accuracy_11 = accuracy_score(y_test_11, y_pred_11)
    test_f1_11 = f1_score(y_test_11, y_pred_11, average="macro")

    y_pred_val_11 = y_pred_val_b.copy()
    for i in range(len(y_val_11)):
        b_probs = list(model_b.predict_proba(X_val_b[i])[0])
        b_probs = b_probs + [b_probs[1]] * 10
        y_probs_temp = model_5.predict_proba(X_val_5[i])[0]
        y_probs = [0] * 12
        y_probs[0] = y_probs_temp[0]
        y_probs[1] = y_probs_temp[1]
        y_probs[2] = y_probs_temp[2]
        y_probs[3] = y_probs_temp[2]
        y_probs[4] = y_probs_temp[2]
        y_probs[5] = y_probs_temp[3]
        y_probs[6] = y_probs_temp[4]
        y_probs[7] = y_probs_temp[3]
        y_probs[8] = y_probs_temp[1]
        y_probs[9] = y_probs_temp[1]
        y_probs[10] = y_probs_temp[4]
        y_probs[11] = y_probs_temp[1]
        eleven_probs = model_11.predict_proba(X_val_11[i])[0]
        final_probs = [a*b*c for a, b, c in zip(b_probs, y_probs, eleven_probs)]
        y_pred_val_11[i] = max(range(len(final_probs)), key=final_probs.__getitem__)
    val_accuracy_11 = accuracy_score(y_val_11, y_pred_val_11)
    val_f1_11 = f1_score(y_val_11, y_pred_val_11, average="macro")

    # Print overall evaluation for 11-way model
    print("\n11-way classification model evaluation on Validation Set:")
    print(f"Validation accuracy: {val_accuracy_11:.4f}")
    print(f"Validation F1 score: {val_f1_11:.4f}")
    print("\n11-way classification model evaluation on Test Set:")
    print(f"Test accuracy: {test_accuracy_11:.4f}")
    print(f"Test F1 score: {test_f1_11:.4f}")

    # if jump_1_11:
    #     # 11-way classification
    #     print(f"\nTraining and evaluating {'11-way'} classification model from 1")
    #     print(f"SMOTE applied: {datasets['11-way']['smote_applied']}")

        # x_train_11, y_train_11 = datasets["11-way"]["train"]
        # x_train_11 = x_train_11[y_train_11 != 0]
        # y_train_11 = y_train_11[y_train_11 != 0]
        # X_val_11, y_val_11 = datasets["11-way"]["val"]
        # x_test_11, y_test_11 = datasets["11-way"]["test"]

    #     x_train_11_resampled, y_train_11_resampled = apply_smote(x_train_11, y_train_11)  # type: ignore

    #     # Train the model
    #     model_11 = LogisticRegression(max_iter=max_iter)
    #     model_11.fit(x_train_11_resampled, y_train_11_resampled)

    #     # initialize an empty numpy array to store the predictions for the 11-way classification
    #     y_pred_11 = y_pred_b.copy()
    #     for i in range(len(y_test_11)):
    #         if y_pred_11[i] != 0:
    #             y_pred_11[i] = model_11.predict(x_test_11[i].reshape(1, -1))[0]
    #     test_accuracy_11 = accuracy_score(y_test_11, y_pred_11)
    #     test_f1_11 = f1_score(y_test_11, y_pred_11, average="macro")

    #     y_pred_val_11 = y_pred_val_b.copy()
    #     for i in range(len(y_val_11)):
    #         if y_pred_val_11[i] != 0:
    #             y_pred_val_11[i] = model_11.predict(X_val_11[i].reshape(1, -1))[0]
    #     val_accuracy_11 = accuracy_score(y_val_11, y_pred_val_11)
    #     val_f1_11 = f1_score(y_val_11, y_pred_val_11, average="macro")

    #     # Print overall evaluation for 11-way model
    #     print("\n11-way classification model evaluation on Validation Set:")
    #     print(f"Validation accuracy: {val_accuracy_11:.4f}")
    #     print(f"Validation F1 score: {val_f1_11:.4f}")
    #     print("\n11-way classification model evaluation on Test Set:")
    #     print(f"Test accuracy: {test_accuracy_11:.4f}")
    #     print(f"Test F1 score: {test_f1_11:.4f}")

    #     return

    # # 11-way classification
    # print(f"\nTraining and evaluating {'11-way'} classification model")
    # print(f"SMOTE applied: {datasets['11-way']['smote_applied']}")

    # # Create separate models for the four categories (1, 2, 3, 4)
    # categories = [1, 2, 3, 4]
    # models = {i: LogisticRegression(max_iter=max_iter) for i in categories}

    # # Prepare datasets
    # x_train_11, y_train_11 = datasets["11-way"]["train"]
    # X_val_11, y_val_11 = datasets["11-way"]["val"]
    # x_test_11, y_test_11 = datasets["11-way"]["test"]

    # reverse_vector_mapping = reverse_dict(vector_mapping)
    # y_train_11_5 = np.array(
    #     [get_5label_from_11label(reverse_vector_mapping[y]) for y in y_train_11]
    # )

    # datasets = {
    #     i: (x_train_11[y_train_11_5 == i], y_train_11[y_train_11_5 == i])
    #     for i in categories
    # }

    # for category in categories:
    #     print(f"Training and evaluating model for category {category}")

    #     x_train_11_i, y_train_11_i = datasets[category]
    #     x_train_11_i_resampled, y_train_11_i_resampled = apply_smote( # type: ignore
    #         x_train_11_i, y_train_11_i
    #     )

    #     # Train the model
    #     models[category].fit(x_train_11_i_resampled, y_train_11_i_resampled)

    # # Evaluate the model
    # y_pred_11 = np.zeros(len(y_pred_b))
    # for i, y in enumerate(y_test_11):
    #     y_5 = y_pred_5[i]
    #     if y_5 != 0:
    #         y_pred_11[i] = models[y_5].predict(x_test_11[i].reshape(1, -1))
    #         # print(y_pred_11[i], y_test_11[i])
    # test_accuracy_11 = accuracy_score(y_test_11, y_pred_11)
    # test_f1_11 = f1_score(y_test_11, y_pred_11, average="macro")

    # y_pred_val_11 = np.zeros(len(y_val_11))
    # for i, y in enumerate(y_val_11):
    #     y_5 = y_pred_val_5[i]
    #     if y_5 != 0:
    #         y_pred_val_11[i] = models[y_5].predict(X_val_11[i].reshape(1, -1))
    # val_accuracy_11 = accuracy_score(y_val_11, y_pred_val_11)
    # val_f1_11 = f1_score(y_val_11, y_pred_val_11, average="macro")

    # # Print overall evaluation for 11-way model
    # print(f"\n{'11-way'} classification model evaluation on Validation Set:")
    # print(f"Validation accuracy: {val_accuracy_11:.4f}")
    # print(f"Validation F1 score: {val_f1_11:.4f}")
    # print(f"\n{'11-way'} classification model evaluation on Test Set:")
    # print(f"Test accuracy: {test_accuracy_11:.4f}")
    # print(f"Test F1 score: {test_f1_11:.4f}")

    return


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Train and evaluate logistic regression models with optional SMOTE sampling."
    )
    parser.add_argument(
        "--use_smote",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Whether to use SMOTE for data sampling (default: True)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="The maximum number of iterations for the logistic regression model (default: 1000)",
    )
    parser.add_argument(
        "--jump_1_11",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Whether to jump from 1 to 11 way classification",
    )
    args = parser.parse_args()

    # Run the main function with the specified SMOTE option
    main(args.use_smote, args.max_iter, args.jump_1_11)
    print("\nTraining and evaluation completed for all tasks.")
