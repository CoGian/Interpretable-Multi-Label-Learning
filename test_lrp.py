import json
import re

from transformers import AutoTokenizer
from LRP_BERT_explainability.BERT.BertForMultiLabelSequenceClassification import BertForMultiLabelSequenceClassification
from LRP_BERT_explainability.ExplanationGenerator import Generator
from sklearn import preprocessing
import numpy as np
import torch
from tqdm import tqdm

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

	for item in tqdm(val_dataset):

		if item["pmid"] != "12204665":
			continue
		text = item["text"].lower()
		text = re.sub(r'(\w)\.', r'\g<1>,', text)
		encoding = tokenizer([text], return_tensors='pt', truncate=True)
		input_ids = encoding['input_ids'].to(device)
		attention_mask = encoding['attention_mask'].to(device)
		gold_labels = mlb.transform([item["labels"]])[0]
		gold_indexes = [i for i, j in enumerate(gold_labels) if j >= 1]

		expl, output, indexes = explanations.generate_LRP(input_ids=input_ids,
												 attention_mask=attention_mask,
												 start_layer=0)
		try:
			for i, output_index in enumerate(indexes):
				if output_index not in gold_indexes:
					continue
				sentences_expl = []
				sent_expl = []
				for index, id in enumerate(input_ids[0]):
					sent_expl.append(expl[i][index].cpu().detach().numpy())
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
					if score > 0.8:
						if gold_label in item["labels_per_sentence"][index]:
							tp += 1
						else:
							fp += 1
					else:
						if gold_label in item["labels_per_sentence"][index]:
							fn += 1
						else:
							tn += 1
		except IndexError:
			print(item["pmid"])

		print("tp", tp)
		print("fp", fp)
		print("tn", tn)
		print("fn", fn)

