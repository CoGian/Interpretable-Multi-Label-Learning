import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing
import re
from nltk import TreebankWordTokenizer
from transformers import AutoTokenizer
import numpy as np


class CsvDataset(Dataset):

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
        text = preprocess_text(text)

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


class JsonDataset(Dataset):

    def __init__(self, json_file, topics_file, tokenizer, max_len, token_classification=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset = pd.read_json(json_file)
        self.token_classification = True
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

        text = row.text.strip()
        text = preprocess_text(text)

        labels = row.labels
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

        item = {
            'ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten(),
            'targets': torch.FloatTensor(labels[0]),
        }

        if self.token_classification:
            sentence_counter = 0
            labels_per_sentence = row.labels_per_sentence
            encoded_label_per_sentence = None
            targets_per_input_id = None
            for index, id in enumerate(inputs['input_ids'].flatten()):
                if index == 0:
                    encoded_label_per_sentence = torch.FloatTensor(
                        self.mlb.transform([labels_per_sentence[sentence_counter]])[0]).unsqueeze(0)
                    targets_per_input_id = encoded_label_per_sentence
                else:
                    targets_per_input_id = torch.cat(
                        (targets_per_input_id, encoded_label_per_sentence))

                if id == 1012:
                    sentence_counter += 1
                    if sentence_counter >= len(labels_per_sentence):
                        encoded_label_per_sentence = torch.FloatTensor(
                            self.mlb.transform([[]])[0]).unsqueeze(0)
                    else:
                        encoded_label_per_sentence = torch.FloatTensor(
                            self.mlb.transform([labels_per_sentence[sentence_counter]])[0]).unsqueeze(0)

            item['targets_per_input_id'] = targets_per_input_id

        return item


def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=1,
                      shuffle=shuffle)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    tokenized = TreebankWordTokenizer().tokenize(text)
    text = ' '.join(tokenized)
    text = re.sub(r"\s's\b", "'s", text)
    text = re.sub(r'(\S)\.', r'\g<1>,', text)
    text = re.sub(r'\.(\S)', r',\g<1>', text)

    return text


if __name__ == '__main__':

    temp_tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12")
    train_dataset = JsonDataset(
        "../../Datasets/HoC/train.json",
        "../../Datasets/HoC/topics.json",
        temp_tokenizer,
        512,
        True
        )
    train_dataloader = get_dataloader(train_dataset, 1, False)

    for batch in train_dataloader:
        print(batch["ids"].shape)
        print(batch["mask"].shape)
        break
