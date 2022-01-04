import argparse
import json
import re
import numpy as np
from transformers import AutoTokenizer
from multi_label_training.src.model import BertForMultiLabelSequenceClassification
from sklearn import preprocessing
from utils.metrics import update_sentence_metrics, print_metrics, calc_output_diff_all_top1
import torch
from tqdm import tqdm

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--threshold',
		'-t',
		help='The threshold of accepting a sentence as rationale',
		default=0.9)

	parser.add_argument(
		'--dataset_name',
		'-dn',
		help='The dataset name for testing',
		default="HoC")

	args = parser.parse_args()
	threshold = float(args.threshold)
	dataset_name = str(args.dataset_name)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = BertForMultiLabelSequenceClassification.from_pretrained(
		dataset_name + "_models/" + dataset_name + "_ncbi_bert_pubmed_multitask/", multi_task=True)
	model.to(device)
	model.eval()
	tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12")
	topics = []

	with open("Datasets/" + dataset_name + "/topics.json", "r") as f:
		for label in f.readlines():
			topics.append(label.strip())
	mlb = preprocessing.MultiLabelBinarizer()
	mlb.fit([topics])

	with open("Datasets/" + dataset_name + "/val.json", "r") as fval:
		val_dataset = json.load(fval)

	scores = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "faithfulness": [], "faithfulness_top1": [], "faithfulness_all_top_1": []}
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

		output = model(input_ids, attention_mask)
		targets_per_input_id = torch.sigmoid(output[1][1])[0]

		output_indexes = [i for i, j in enumerate(torch.sigmoid(output[0].logits).cpu().detach().numpy()[0]) if j >= .5]
		top_sent_per_label = []

		for output_index in output_indexes:
			if output_index not in gold_indexes:
				continue

			sentences_expl = []
			sent_expl = []

			for index, id in enumerate(input_ids[0]):
				sent_expl.append(targets_per_input_id[index, output_index].cpu().detach().numpy())
				if id.cpu().detach().numpy() == 1012:
					sentences_expl.append(sent_expl)
					sent_expl = []

			update_sentence_metrics(
				sentences_expl,
				gold_labels,
				output_index,
				output[0],
				scores,
				scores_per_label,
				top_sent_per_label,
				mlb,
				item,
				text,
				model,
				tokenizer,
				threshold
			)

		output_indexes = [output_index for output_index in output_indexes if output_index in gold_indexes]
		if top_sent_per_label:
			scores["faithfulness_all_top_1"].append(calc_output_diff_all_top1(output[0], output_indexes, text, top_sent_per_label, model, tokenizer))

	scores["faithfulness"] = np.mean(scores["faithfulness"])
	scores["faithfulness_top1"] = np.mean(scores["faithfulness_top1"])
	scores["faithfulness_all_top_1"] = np.mean(scores["faithfulness_all_top_1"])
	print_metrics(scores, scores_per_label, topics)
