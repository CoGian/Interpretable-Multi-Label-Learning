import argparse
import json
import re

from transformers import AutoTokenizer
from multi_label_training.src.model import BertForMultiLabelSequenceClassification
from RAW_ATTN_explainablity.explainer import Explainer
from sklearn import preprocessing
from utils.metrics import update_sentence_metrics, print_metrics
import torch
from tqdm import tqdm

if __name__ == '__main__':

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = BertForMultiLabelSequenceClassification.from_pretrained("HoC_models/HoC_ncbi_bert_pubmed_multitask/",
																	multi_task=True)
	model.to(device)
	model.eval()
	tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12")
	topics = []
	with open("Datasets/HoC/topics.json", "r") as f:
		for label in f.readlines():
			topics.append(label.strip())
	mlb = preprocessing.MultiLabelBinarizer()
	mlb.fit([topics])

	with open("Datasets/HoC/val.json", "r") as fval:
		val_dataset = json.load(fval)

	for item in tqdm(val_dataset):

		text = item["text"].lower()
		text = re.sub(r'(\S)\.', r'\g<1>,', text)
		text = re.sub(r'\.(\S)', r',\g<1>', text)
		encoding = tokenizer([text], return_tensors='pt', max_length=512, truncation=True)
		input_ids = encoding['input_ids'].to(device)
		attention_mask = encoding['attention_mask'].to(device)

		output = model(input_ids, attention_mask)

		print(output)