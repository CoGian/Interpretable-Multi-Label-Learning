import numpy as np
import pandas as pd
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def min_max_scaling(a,b, vector):
	return a + (((vector - vector.min()) * (b - a)) / (vector.max() - vector.min()))


def max_abs_scaling(vector):
	return vector / np.abs(vector).max()


def update_sentence_metrics(
		sentences_expl,
		gold_labels,
		output_index,
		output,
		scores,
		scores_per_label,
		top_sent_per_label,
		mlb,
		item,
		text,
		model,
		tokenizer,
		threshold=0.9):

	sent_scores = []
	for sent_expl in sentences_expl:
		sent_scores.append(np.mean(np.sort(sent_expl)[-10:]))

	sent_scores = np.array(sent_scores)

	scaled_sent_scores = min_max_scaling(0, 1, sent_scores)
	one_hot = np.zeros((1, len(gold_labels)), dtype=np.float32)
	one_hot[0, output_index] = 1

	gold_label = mlb.inverse_transform(one_hot)[0][0]
	logit_output = torch.sigmoid(output[0][0])[output_index].cpu().detach()

	pred_pos_sentences = []
	for index, score in enumerate(scaled_sent_scores):
		if score > threshold:
			if gold_label.lower() in item["labels_per_sentence"][index]:
				scores['tp'] += 1
				scores_per_label[gold_label]["tp"] += 1
			else:
				scores['fp'] += 1
				scores_per_label[gold_label]["fp"] += 1
			pred_pos_sentences.append(index)
		else:
			if gold_label.lower() in item["labels_per_sentence"][index]:
				scores['fn'] += 1
				scores_per_label[gold_label]["fn"] += 1
			else:
				scores['tn'] += 1
				scores_per_label[gold_label]["tn"] += 1

	if pred_pos_sentences:
		output_diff = calc_output_diff(logit_output, output_index, text, pred_pos_sentences, model, tokenizer)
		top_sent_per_label.append(np.sort(pred_pos_sentences)[-1])
		output_diff_top1 = calc_output_diff(logit_output, output_index, text, np.sort(pred_pos_sentences)[-1:], model, tokenizer)
		scores["faithfulness"].append(output_diff)
		scores["faithfulness_top1"].append(output_diff_top1)


def print_metrics(scores, scores_per_label, topics):
	print("tp", scores['tp'])
	print("fp", scores['fp'])
	print("tn", scores['tn'])
	print("fn", scores['fn'])
	recall = scores['tp'] / (scores['tp'] + scores['fn'])
	precision = scores['tp'] / (scores['tp'] + scores['fp'])
	print("Recall: ", recall)
	print("Precision: ", precision)
	print("F1: ", (2 * recall * precision) / (recall + precision))
	print("Faithfulness: ", scores["faithfulness"])

	metrics_per_labels = {}
	for label in topics:
		recall = scores_per_label[label]["tp"] / (scores_per_label[label]["tp"] + scores_per_label[label]["fn"])
		precision = scores_per_label[label]["tp"] / (scores_per_label[label]["tp"] + scores_per_label[label]["fp"])
		f1 = (2 * recall * precision) / (recall + precision)
		metrics_per_labels[label] = {"recall": recall, "precision": precision, "f1": f1}

	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(pd.DataFrame(metrics_per_labels))


def calc_output_diff(logit_output, output_index, text, sentence_indexes, model, tokenizer, all_top_1=False):

	text_sentences = text.split(".")
	text_sentences = [sent for index, sent in enumerate(text_sentences) if index not in sentence_indexes]
	input_text = " . ".join(text_sentences)

	encoding = tokenizer([input_text], return_tensors='pt', max_length=512, truncation=True)
	input_ids = encoding['input_ids'].to(device)
	attention_mask = encoding['attention_mask'].to(device)

	try:
		if model.multi_task:
			logit_perturbed = torch.sigmoid(model(input_ids, attention_mask)[0][0][0])[output_index].cpu().detach()
		else:
			logit_perturbed = torch.sigmoid(model(input_ids, attention_mask)[0][0])[output_index].cpu().detach()
	except AttributeError:
		logit_perturbed = torch.sigmoid(model(input_ids, attention_mask)[0][0])[output_index].cpu().detach()

	diff = float(logit_output - logit_perturbed)

	return diff


def calc_output_diff_all_top1(logit_outputs, output_indexes, text, sentence_indexes, model, tokenizer):

	text_sentences = text.split(".")
	text_sentences = [sent for index, sent in enumerate(text_sentences) if index not in sentence_indexes]
	input_text = " . ".join(text_sentences)

	encoding = tokenizer([input_text], return_tensors='pt', max_length=512, truncation=True)
	input_ids = encoding['input_ids'].to(device)
	attention_mask = encoding['attention_mask'].to(device)

	try:
		if model.multi_task:
			perturbed_output = torch.sigmoid(model(input_ids, attention_mask)[0][0][0]).cpu().detach()
		else:
			perturbed_output = torch.sigmoid(model(input_ids, attention_mask)[0][0]).cpu().detach()
	except AttributeError:
		perturbed_output = torch.sigmoid(model(input_ids, attention_mask)[0][0]).cpu().detach()

	diff = 0

	for output_index in output_indexes:
		logit_output = torch.sigmoid(logit_outputs[0][0])[output_index].cpu().detach()
		logit_perturbed = perturbed_output[output_index]
		diff += float(logit_output - logit_perturbed)

	diff /= len(output_indexes)
	return diff

