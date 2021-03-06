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
from utils.metrics import update_sentence_metrics, print_metrics, max_abs_scaling, min_max_scaling, \
	calc_output_diff_all_top1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

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

	parser.add_argument(
		'--most_important_tokens',
		'-mt',
		help='The most important tokens per sentence to kane into consideration: 0:all',
		default=10)

	parser.add_argument(
		'--model_mode',
		'-m',
		help='The model used for tests: simple or multi',
		default="simple")

	args = parser.parse_args()
	weight_aggregation = str(args.weight_aggregation)
	dataset_name = str(args.dataset_name)
	most_important_tokens = int(args.most_important_tokens)
	model_mode = str(args.model_mode)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

	if model_mode == "multi":
		model = BertForMultiLabelSequenceClassification.from_pretrained(
			dataset_name + "_models/" + dataset_name + "_bert_2_multi_task/")
	else:
		model = BertForMultiLabelSequenceClassification.from_pretrained(
			dataset_name + "_models/" + dataset_name + "_bert/")

	model.to(device)
	model.eval()

	explanations = Generator(model, weight_aggregation=weight_aggregation)

	topics = []
	with open("Datasets/" + dataset_name + "/topics.json", "r") as f:
		for label in f.readlines():
			topics.append(label.strip())
	mlb = preprocessing.MultiLabelBinarizer()
	mlb.fit([topics])

	with open("Datasets/" + dataset_name + "/test.json", "r") as fval:
		val_dataset = json.load(fval)

	scores_per_threshold = [{"tp": 0, "fp": 0, "tn": 0, "fn": 0, "faithfulness": [], "faithfulness_top1": [],
							 "faithfulness_all_top_1": []} for threshold in range(9, 0, -1)]
	scores_per_label_per_threshold = [{label: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for label in topics} for threshold in
									  range(9, 0, -1)]
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

		except RuntimeError:
			print("RuntimeError error for item: ", item["pmid"])
			continue

		for threshold_index, threshold in enumerate(range(9, 0, -1)):

			top_sent_per_label = []

			try:
				for i, output_index in enumerate(output_indexes):
					if output_index not in gold_indexes:
						continue

					sentences_expl = []
					sent_expl = []
					for index, id in enumerate(input_ids[0]):
						sent_expl.append(word_attributions_per_pred_class[i][index])
						if id.cpu().detach().numpy() == 1012:
							sentences_expl.append(sent_expl)
							sent_expl = []

					scores, scores_per_label = update_sentence_metrics(
						sentences_expl,
						gold_labels,
						output_index,
						output,
						scores_per_threshold[threshold_index],
						scores_per_label_per_threshold[threshold_index],
						top_sent_per_label,
						mlb,
						item,
						text,
						model,
						tokenizer,
						threshold / 10,
						most_important_tokens
					)

					scores_per_threshold[threshold_index] = scores
					scores_per_label_per_threshold[threshold_index] = scores_per_label

				output_indexes = [output_index for output_index in output_indexes if output_index in gold_indexes]
				if top_sent_per_label:
					scores_per_threshold[threshold_index]["faithfulness_all_top_1"].append(
						calc_output_diff_all_top1(output, output_indexes, text, top_sent_per_label, model, tokenizer))

			except IndexError:
				print("IndexError error for item: ", item["pmid"])
				continue

	for threshold_index, threshold in enumerate(range(9, 0, -1)):
		print("Results for threshold: ", threshold / 10)
		scores_per_threshold[threshold_index]["faithfulness"] = np.mean(
			scores_per_threshold[threshold_index]["faithfulness"])
		scores_per_threshold[threshold_index]["faithfulness_top1"] = np.mean(
			scores_per_threshold[threshold_index]["faithfulness_top1"])
		scores_per_threshold[threshold_index]["faithfulness_all_top_1"] = np.mean(
			scores_per_threshold[threshold_index]["faithfulness_all_top_1"])
		print_metrics(scores_per_threshold[threshold_index], scores_per_label_per_threshold[threshold_index], topics)
