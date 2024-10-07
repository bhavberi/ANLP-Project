# ANLP-Project

This project implements various machine learning models for classifying text as sexist or not, using different classification schemes (binary, 5-way, and 11-way).

## File Structure

1. `clean_data.py`: Contains functions for data preprocessing, cleaning, and preparation.
2. `linear_regression.py`: Implements logistic regression models for the classification tasks.
3. `transformer.py`: Implements BERT-based transformer models for the classification tasks.

## Setup and Execution

1. Install the required dependencies:

```pip install -r requirements.txt```

2. Ensure you have the dataset file `edos_labelled_aggregated.csv` in the same directory as the scripts.

3. Run the logistic regression models:

```python linear_regression.py [--use_smote True/False]```

4. Run the transformer model:

```python transformer.py [--use_smote True/False]```


The `--use_smote` flag is optional and defaults to True. Set it to False if you don't want to use SMOTE for data sampling.

## Notes

- The scripts will automatically download required NLTK data to a local `nltk_data` directory.
- Best models for each task in the transformer approach will be saved as `best_model_{task}.pth`.
- Both scripts will print out performance metrics for each classification task.
