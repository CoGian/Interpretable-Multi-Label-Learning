import json
import re

from transformers import AutoTokenizer
from LRP_BERT_explainability.BERT.BertForMultiLabelSequenceClassification import BertForMultiLabelSequenceClassification
from LRP_BERT_explainability.ExplanationGenerator import Generator
from sklearn import preprocessing
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

if __name__ == '__main__':

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model = BertForMultiLabelSequenceClassification.from_pretrained("HoC_models/HoC_ncbi_bert_pubmed/")
	model.to(device)
	model.eval()

	tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12")

	explanations = Generator(model)

	topics = []
	with open("Datasets/HoC/topics.json", "r") as f:
		for label in f.readlines():
			topics.append(label.strip())
	mlb = preprocessing.MultiLabelBinarizer()
	mlb.fit([topics])

	with open("Datasets/HoC/val.json", "r") as fval:
		val_dataset = json.load(fval)

	tp = 0
	fp = 0
	tn = 0
	fn = 0
	scores_per_label = {label: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for label in topics}
	for item in tqdm(val_dataset):

		text = item["text"].lower()
		text = re.sub(r'(\S)\.', r'\g<1>,', text)
		text = re.sub(r'\.(\S)', r',\g<1>', text)
		encoding = tokenizer([text], return_tensors='pt', max_length=512, truncation=True)
		input_ids = encoding['input_ids'].to(device)
		attention_mask = encoding['attention_mask'].to(device)
		gold_labels = mlb.transform([item["labels"]])[0]
		gold_indexes = [i for i, j in enumerate(gold_labels) if j >= 1]

		word_attributions, output, output_indexes = explanations.generate_LRP(input_ids=input_ids,
																			  attention_mask=attention_mask,
																			  start_layer=0)
		try:
			for i, output_index in enumerate(output_indexes):
				if output_index not in gold_indexes:
					continue
				sentences_expl = []
				sent_expl = []
				for index, id in enumerate(input_ids[0]):
					sent_expl.append(word_attributions[i][index].cpu().detach().numpy())
					if id.cpu().detach().numpy() == 1012:
						sentences_expl.append(sent_expl)
						sent_expl = []
				sent_scores = []
				for sent_expl in sentences_expl:
					sent_scores.append(np.mean(sent_expl))

				sent_scores = np.array(sent_scores)
				sent_scores = (sent_scores - sent_scores.min()) / (sent_scores.max() - sent_scores.min())

				one_hot = np.zeros((1, len(gold_labels)), dtype=np.float32)
				one_hot[0, output_index] = 1

				gold_label = mlb.inverse_transform(one_hot)[0][0]

				for index, score in enumerate(sent_scores):
					if score > 0.9:
						if gold_label in item["labels_per_sentence"][index]:
							tp += 1
							scores_per_label[gold_label]["tp"] += 1
						else:
							fp += 1
							scores_per_label[gold_label]["fp"] += 1
					else:
						if gold_label in item["labels_per_sentence"][index]:
							fn += 1
							scores_per_label[gold_label]["fn"] += 1
						else:
							tn += 1
							scores_per_label[gold_label]["tn"] += 1
		except IndexError:
			print("4444 error for item: ", item["pmid"])

	print("tp", tp)
	print("fp", fp)
	print("tn", tn)
	print("fn", fn)
	recall = tp / (tp+fn)
	precision = tp / (tp+fp)
	print("Recall: ", recall)
	print("Precision: ", precision)
	print("F1: ", (2*recall*precision)/(recall+precision)) #micro

	metrics_per_labels = {}
	for label in topics:
		recall = scores_per_label[label]["tp"] / (scores_per_label[label]["tp"] + scores_per_label[label]["fn"])
		precision = scores_per_label[label]["tp"] / (scores_per_label[label]["tp"] + scores_per_label[label]["fp"])
		f1 = (2*recall*precision)/(recall+precision)
		metrics_per_labels[label] = {"recall": recall, "precision": precision, "f1": f1}

	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(pd.DataFrame(metrics_per_labels))


