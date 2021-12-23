import argparse
import json
import re

import numpy as np
from transformers import AutoTokenizer
from LRP_BERT_explainability.BERT.BertForMultiLabelSequenceClassification import BertForMultiLabelSequenceClassification
from LRP_BERT_explainability.ExplanationGenerator import Generator
from sklearn import preprocessing
import torch
from tqdm import tqdm
from utils.metrics import update_sentence_metrics, print_metrics, max_abs_scaling, min_max_scaling

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--threshold',
		'-t',
		help='The threshold of accepting a sentence as rationale',
		default=0.9)
	parser.add_argument(
		'--weight_aggregation',
		'-wa',
		help='The weight aggregation methods: mean, mean_pos, mean_abs',
		default="mean")

	parser.add_argument(
		'--dataset_name',
		'-dn',
		help='The dataset name for testing',
		default="HoC")
	args = parser.parse_args()
	threshold = float(args.threshold)
	weight_aggregation = str(args.weight_aggregation)
	dataset_name = str(args.dataset_name)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model = BertForMultiLabelSequenceClassification.from_pretrained(
		dataset_name + "_models/" + dataset_name + "_ncbi_bert_pubmed/")
	model.to(device)
	model.eval()

	tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12")

	explanations = Generator(model, weight_aggregation=weight_aggregation)

	topics = []
	with open("Datasets/" + dataset_name + "/topics.json", "r") as f:
		for label in f.readlines():
			topics.append(label.strip())
	mlb = preprocessing.MultiLabelBinarizer()
	mlb.fit([topics])

	with open("Datasets/" + dataset_name + "/val.json", "r") as fval:
		val_dataset = json.load(fval)

	scores = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "faithfulness": 0}
	scores_per_label = {label: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for label in topics}
	count_output_expl = 0
	for item in tqdm(val_dataset):

		text = item["text"].lower()
		text = re.sub(r'(\S)\.', r'\g<1>,', text)
		text = re.sub(r'\.(\S)', r',\g<1>', text)
		encoding = tokenizer([text], return_tensors='pt', max_length=512, truncation=True)
		input_ids = encoding['input_ids'].to(device)
		attention_mask = encoding['attention_mask'].to(device)
		gold_labels = mlb.transform([item["labels"]])[0]
		gold_indexes = [i for i, j in enumerate(gold_labels) if j >= 1]

		try:
			word_attributions_per_pred_class, output, output_indexes = explanations.generate_LRP(
				input_ids=input_ids,
				attention_mask=attention_mask,
				start_layer=0)

			word_attributions_per_pred_class = np.array([
				word_attributions.cpu().detach().numpy()
				for word_attributions in word_attributions_per_pred_class])

			word_attributions_per_pred_class = [
				max_abs_scaling(word_attributions) if weight_aggregation == "mean"
				else min_max_scaling(0, 1, word_attributions)
				for word_attributions in word_attributions_per_pred_class]

		except RuntimeError:
			print("RuntimeError error for item: ", item["pmid"])
			continue

		try:
			for i, output_index in enumerate(output_indexes):
				if output_index not in gold_indexes:
					continue

				count_output_expl += 1
				sentences_expl = []
				sent_expl = []
				for index, id in enumerate(input_ids[0]):
					sent_expl.append(word_attributions_per_pred_class[i][index])
					if id.cpu().detach().numpy() == 1012:
						sentences_expl.append(sent_expl)
						sent_expl = []

				update_sentence_metrics(
					sentences_expl,
					gold_labels,
					output_index,
					output,
					scores,
					scores_per_label,
					mlb,
					item,
					text,
					model,
					tokenizer,
					threshold
				)

		except IndexError:
			print("IndexError error for item: ", item["pmid"])
			continue

	scores["faithfulness"] /= count_output_expl
	print_metrics(scores, scores_per_label, topics)
