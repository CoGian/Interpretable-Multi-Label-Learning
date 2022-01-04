import argparse
import json
import re

from transformers import AutoTokenizer
from multi_label_training.src.model import BertForMultiLabelSequenceClassification
from sklearn import preprocessing
from utils.metrics import update_sentence_metrics, print_metrics
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

		output = model(input_ids, attention_mask)
		targets_per_input_id = torch.sigmoid(output[1][1])[0]

		output_indexes = [i for i, j in enumerate(torch.sigmoid(output[0].logits).cpu().detach().numpy()[0]) if j >= .5]

		for output_index in output_indexes:
			if output_index not in gold_indexes:
				continue

			count_output_expl += 1
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
				mlb,
				item,
				text,
				model,
				tokenizer,
				threshold
			)

	scores["faithfulness"] /= count_output_expl
	print_metrics(scores, scores_per_label, topics)
