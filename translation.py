import argparse
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Translate text")
parser.add_argument(
    "--to", type=str, default="hin_Deva", help="Language code to translate to"
)
parser.add_argument(
    "--fullname", type=str, default="hindi", help="Full name of the language"
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = (
    AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")
    .to("cuda")
    .eval()
)

df = pd.read_csv("./edos_labelled_aggregated.csv")

# Define batch size
batch_size = 8  # Adjust based on your GPU memory capacity


def batch_translation(batch_texts):
    inputs = tokenizer(
        batch_texts, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.to),
        max_length=70,
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


# DataLoader to handle batching
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


# Create DataLoader
text_dataset = TextDataset(df["text"].tolist())
data_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False)

# Run batched translation and collect results
translated_texts = []
for batch in tqdm(data_loader):
    translated_batch = batch_translation(batch)
    translated_texts.extend(translated_batch)

# Store the results in the DataFrame
df["translated_text"] = translated_texts

# Save the translated data
df.to_csv(f"./edos_labelled_aggregated_translated_{args.fullname}.csv", index=False)
