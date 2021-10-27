import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing
import re
from nltk import TreebankWordTokenizer
from transformers import AutoTokenizer


class LitCovidDataset(Dataset):

    def __init__(self, csv_file, topics_file, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset = pd.read_csv(csv_file)
        self.topics = []
        with open(topics_file, "r") as f:
            for label in f.readlines():
                self.topics.append(label.strip())
        self.mlb = preprocessing.MultiLabelBinarizer()
        self.mlb.fit([self.topics])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset.iloc[index]

        title = row.title.strip()
        abstract = row.abstract.strip()
        text = title + ". " + abstract
        text = text.lower()
        text = re.sub(r'[\r\n]+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        tokenized = TreebankWordTokenizer().tokenize(text)
        text = ' '.join(tokenized)
        text = re.sub(r"\s's\b", "'s", text)

        labels = row.label.split(";")
        labels = self.mlb.transform([labels])

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten(),
            'targets': torch.FloatTensor(labels[0])
        }


def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=1,
                      shuffle=shuffle)


if __name__ == '__main__':

    temp_tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12")
    train_dataset = LitCovidDataset(
        "../../Datasets/LitCovid/BC7-LitCovid-Train.csv",
        "../../Datasets/LitCovid/topics.json",
        temp_tokenizer,
        248
        )
    train_dataloader = get_dataloader(train_dataset, 2, False)

    for batch in train_dataloader:
        print(batch["ids"].shape)
        print(batch["mask"].shape)
        print(batch["targets"].shape)
        print(batch["targets"])
        break
