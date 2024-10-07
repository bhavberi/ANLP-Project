import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('wordnet', quiet=True, download_dir='./nltk_data/')
nltk.download('punkt', quiet=True, download_dir='./nltk_data/')
nltk.data.path.append('./nltk_data/')

def clean_text(df):
    lemmatizer = WordNetLemmatizer()
    data = list(zip(df['text'], df['label_sexist'], df['label_category'], df['label_vector'], df['split']))
    val_data, train_data, test_data = [], [], []
    
    for counter, (sentence, label_sexist, label_category, label_vector, split) in enumerate(tqdm(data)):
        # Clean and preprocess the text
        sentence = sentence.strip().lower()
        sentence = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', '', sentence)  # Remove URLs
        sentence = re.sub(r'\[(URL|USER)\]', '', sentence)  # Remove [URL] and [USER] tags
        tokenized_words = word_tokenize(sentence)
        tokenized_words = [re.sub(r'[^a-z0-9]', '', word) for word in tokenized_words if word]  # Keep only alphanumeric characters
        tokenized_words = [lemmatizer.lemmatize(word) for word in tokenized_words]  # Lemmatize words
        
        if not tokenized_words:
            print(f"Empty sentence at index {counter}: {df.iloc[counter]}")
            continue
        
        # Assign data to appropriate split
        if split == 'dev':
            val_data.append((tokenized_words, label_sexist, label_category, label_vector))
        elif split == 'train':
            train_data.append((tokenized_words, label_sexist, label_category, label_vector))
        elif split == 'test':
            test_data.append((tokenized_words, label_sexist, label_category, label_vector))
    
    return train_data, val_data, test_data

class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]

def create_label_mappings(df):
    category_mapping = {cat: i for i, cat in enumerate(df['label_category'].unique())}
    vector_mapping = {vec: i for i, vec in enumerate(df['label_vector'].unique())}
    return category_mapping, vector_mapping

def prepare_data(data, task='binary', category_mapping=None, vector_mapping=None):
    texts = np.array([' '.join(sentence) for sentence, *_ in data])
    
    if task == 'binary':
        labels = np.array([1 if label == "sexist" else 0 for _, label, *_ in data])
    elif task == '5-way':
        labels = np.array([category_mapping[label] for _, _, label, _ in data])
    elif task == '11-way':
        labels = np.array([vector_mapping[label] for _, _, _, label in data])
    else:
        raise ValueError("Invalid task. Choose 'binary', '5-way', or '11-way'.")
    
    return texts, labels

def process_data(csv_path, use_smote=True):
    # Load and preprocess data
    df = pd.read_csv(csv_path)
    category_mapping, vector_mapping = create_label_mappings(df)
    train_data, val_data, test_data = clean_text(df)

    # Prepare datasets for each classification task
    tasks = ['binary', '5-way', '11-way']
    datasets = {}

    for task in tasks:
        # Prepare data for current task
        train_texts, train_labels = prepare_data(train_data, task, category_mapping, vector_mapping)
        val_texts, val_labels = prepare_data(val_data, task, category_mapping, vector_mapping)
        test_texts, test_labels = prepare_data(test_data, task, category_mapping, vector_mapping)
        
        # Initialize and fit TfidfVectorizer
        vectorizer = TfidfVectorizer(tokenizer=Lemmatizer(), lowercase=False, token_pattern=None)
        train_texts_vectorized = vectorizer.fit_transform(train_texts)
        val_texts_vectorized = vectorizer.transform(val_texts)
        test_texts_vectorized = vectorizer.transform(test_texts)
        
        # Apply SMOTE for handling class imbalance if use_smote is True
        if use_smote:
            smote = SMOTE(sampling_strategy='not majority')
            train_texts_resampled, train_labels_resampled = smote.fit_resample(train_texts_vectorized, train_labels)
        else:
            train_texts_resampled, train_labels_resampled = train_texts_vectorized, train_labels
        
        # Store prepared datasets
        datasets[task] = {
            'train': (train_texts_resampled, train_labels_resampled),
            'val': (val_texts_vectorized, val_labels),
            'test': (test_texts_vectorized, test_labels),
            'smote_applied': use_smote
        }

    return datasets, category_mapping, vector_mapping

if __name__ == "__main__":
    # This block will only run if clean_data.py is executed directly
    csv_path = 'edos_labelled_aggregated.csv'  # Update this path as needed
    datasets, category_mapping, vector_mapping = process_data(csv_path, use_smote=True)
    
    # Print dataset information
    print("Datasets prepared for the following tasks:")
    for task in datasets:
        print(f"- {task} classification")
        print(f"  Train shape: {datasets[task]['train'][0].shape}")
        print(f"  Validation shape: {datasets[task]['val'][0].shape}")
        print(f"  Test shape: {datasets[task]['test'][0].shape}")
        print(f"  Number of classes: {len(np.unique(datasets[task]['train'][1]))}")
        print(f"  SMOTE applied: {datasets[task]['smote_applied']}")

        unique_labels, counts = np.unique(datasets[task]['train'][1], return_counts=True)
        print("  Category-wise shapes after resampling:")
        for label, count in zip(unique_labels, counts):
            if task == 'binary':
                category = "sexist" if label == 1 else "not sexist"
            elif task == '5-way':
                category = list(category_mapping.keys())[list(category_mapping.values()).index(label)]
            else:  # 11-way
                category = list(vector_mapping.keys())[list(vector_mapping.values()).index(label)]
            print(f"    {category}: {count}")
        
        print()
