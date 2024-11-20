import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm
nltk.data.path.append("./nltk_data/")

# Load the data
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Process the data
def clean_data(inputs):
    cleaned_inputs = []
    counter = 0
    for sentence in tqdm(inputs):
        lemmatizer = WordNetLemmatizer()
        sentence = sentence.strip().lower()
        sentence = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', "", sentence)  # Remove URLs
        sentence = re.sub(r"\[(url|user)\]", "", sentence)  # Remove [URL] and [USER] tags
        tokenized_words = word_tokenize(sentence)
        tokenized_words = [
            re.sub(r"[^a-z0-9]", "", word) for word in tokenized_words if word
        ]  # Keep only alphanumeric characters
        tokenized_words = [
            lemmatizer.lemmatize(word) for word in tokenized_words
        ]  # Lemmatize words
        tokenized_words = [
            word for word in tokenized_words if word
        ]
        if not tokenized_words:
            print(f"Empty sentence at index {counter}: {inputs[counter]}")
            counter += 1
            continue
        counter += 1
        new_sentence = " ".join(tokenized_words)
        cleaned_inputs.append(new_sentence)
    return cleaned_inputs

def annotate_data(data, female_words):
    annotated_data = []
    for sentence in tqdm(data):
        if any(word in sentence for word in female_words):
            annotated_data.append(1)
        else:
            annotated_data.append(0)
    return annotated_data

data = load_data("edos_labelled_aggregated.csv")
inputs = list(data["text"])
cleaned_inputs = clean_data(inputs)
# female_words = ["she", "her", "hers", "woman", "women", "girl", "lady", "ladies"]
male_words = ["he", "his", "man", "men", "boy", "gentleman", "gentlemen", "bro", "male", "guy"]
annotated_data = annotate_data(cleaned_inputs, male_words)

# Count the number of positive and negative samples
positive_samples = sum(annotated_data)
negative_samples = len(annotated_data) - positive_samples
print(f"Number of positive samples: {positive_samples}")
print(f"Number of negative samples: {negative_samples}")

data_to_save = []
for sentence, label in zip(cleaned_inputs, annotated_data):
    data_to_save.append([sentence, label])

df = pd.DataFrame(data_to_save, columns=["text", "label"])
df.to_csv("male_annotated_data.csv", index=False)