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
		'--dataset_name',
		'-dn',
		help='The dataset name for testing',
		default="HoC")

	parser.add_argument(
		'--most_important_tokens',
		'-mt',
		help='The most important tokens per sentence to kane into consideration: 0:all',
		default=10)

	args = parser.parse_args()
	dataset_name = str(args.dataset_name)
	most_important_tokens = int(args.most_important_tokens)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = BertForMultiLabelSequenceClassification.from_pretrained(
		dataset_name + "_models/" + dataset_name + "_bert_2_multi_task/", multi_task=True)
	model.to(device)
	model.eval()
	tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
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

		output = model(input_ids, attention_mask)
		targets_per_input_id = torch.sigmoid(output[1][1])[0]

		output_indexes = [i for i, j in enumerate(torch.sigmoid(output[0].logits).cpu().detach().numpy()[0]) if j >= .5]

		for threshold_index, threshold in enumerate(range(9, 0, -1)):
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

				scores, scores_per_label = update_sentence_metrics(
						sentences_expl,
						gold_labels,
						output_index,
						output[0],
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
					calc_output_diff_all_top1(output[0], output_indexes, text, top_sent_per_label, model, tokenizer))

	for threshold_index, threshold in enumerate(range(9, 0, -1)):
		print("Results for threshold: ", threshold / 10)
		scores_per_threshold[threshold_index]["faithfulness"] = np.mean(
			scores_per_threshold[threshold_index]["faithfulness"])
		scores_per_threshold[threshold_index]["faithfulness_top1"] = np.mean(
			scores_per_threshold[threshold_index]["faithfulness_top1"])
		scores_per_threshold[threshold_index]["faithfulness_all_top_1"] = np.mean(
			scores_per_threshold[threshold_index]["faithfulness_all_top_1"])
		print_metrics(scores_per_threshold[threshold_index], scores_per_label_per_threshold[threshold_index], topics)


