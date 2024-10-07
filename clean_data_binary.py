# Import necessary libraries
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
nltk.download('wordnet', download_dir='./nltk_data/')
nltk.data.path.append('./nltk_data/')

# Define text cleaning function
def clean_text(df):
    lemmatizer = WordNetLemmatizer()
    data = list(zip(df['text'], df['label_sexist'], df['label_category'], df['label_vector'], df['split']))
    val_data, train_data, test_data = [], [], []
    
    for counter, (sentence, label_sexist, label_category, label_vector, split) in enumerate(tqdm(data)):
        # Clean and preprocess the text
        sentence = sentence.strip().lower()
        sentence = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', '', sentence)
        sentence = re.sub(r'\[(URL|USER)\]', '', sentence)
        tokenized_words = word_tokenize(sentence)
        tokenized_words = [re.sub(r'[^a-z0-9]', '', word) for word in tokenized_words if word]
        tokenized_words = [lemmatizer.lemmatize(word) for word in tokenized_words]
        
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

# Define custom lemmatizer for TfidfVectorizer
class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]

# Load and preprocess data
df = pd.read_csv('edos_labelled_aggregated.csv')
pd.set_option('display.max_colwidth', None)
train_data, val_data, test_data = clean_text(df)

# Combine all data for potential future use
all_data = train_data + val_data + test_data

# Print sample data from each split
print("Sample train data:", train_data[0][0])
print("Sample validation data:", val_data[0][0])
print("Sample test data:", test_data[0][0])
print("Data split sizes:", len(train_data), len(val_data), len(test_data))

# Prepare text and labels for model
def prepare_data(data):
    texts = np.array([' '.join(sentence) for sentence, *_ in data])
    labels = np.array([1 if label == "sexist" else 0 for _, label, *_ in data])
    return texts, labels

train_texts, train_labels = prepare_data(train_data)
val_texts, val_labels = prepare_data(val_data)
test_texts, test_labels = prepare_data(test_data)

# Initialize and fit TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=Lemmatizer(), lowercase=False)
train_texts_vectorized = vectorizer.fit_transform(train_texts)

# Apply SMOTE for handling class imbalance
smote = SMOTE(sampling_strategy='not majority')
train_texts_resampled, train_labels_resampled = smote.fit_resample(train_texts_vectorized, train_labels)
