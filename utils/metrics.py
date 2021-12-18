import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
max_abs_scaler = MaxAbsScaler()


def update_sentence_metrics(
		sentences_expl,
		gold_labels,
		output_index,
		scores,
		scores_per_label,
		mlb,
		item,
		threshold=0.9,
		weight_aggregation="mean"):

	sent_scores = []
	for sent_expl in sentences_expl:
		if weight_aggregation == "mean_pos":
			sent_expl = np.where(sent_expl >= .0, sent_expl, .0)
		elif weight_aggregation == "mean_abs":
			sent_expl = np.abs(sent_expl)
		sent_scores.append(np.mean(max_abs_scaler.fit_transform(np.reshape(sent_expl, (1, -1)))[0]))

	sent_scores = np.array(sent_scores)

	scaled_sent_scores = (sent_scores - sent_scores.min()) / (sent_scores.max() - sent_scores.min())
	# percentile_threshold = np.percentile(sent_scores, threshold)
	one_hot = np.zeros((1, len(gold_labels)), dtype=np.float32)
	one_hot[0, output_index] = 1

	gold_label = mlb.inverse_transform(one_hot)[0][0]

	# gold_label_check = gold_label.split("---")[0] + "--" + gold_label.split("---")[1]
	for index, score in enumerate(scaled_sent_scores):
		if score > threshold:
			if gold_label.lower() in item["labels_per_sentence"][index].lower():
				scores['tp'] += 1
				scores_per_label[gold_label]["tp"] += 1
			else:
				scores['fp'] += 1
				scores_per_label[gold_label]["fp"] += 1
		else:
			if gold_label.lower() in item["labels_per_sentence"][index].lower():
				scores['fn'] += 1
				scores_per_label[gold_label]["fn"] += 1
			else:
				scores['tn'] += 1
				scores_per_label[gold_label]["tn"] += 1


def print_metrics(scores, scores_per_label, topics):
	print("tp", scores['tp'])
	print("fp", scores['fp'])
	print("tn", scores['tn'])
	print("fn", scores['fn'])
	recall = scores['tp'] / (scores['tp'] + scores['fn'])
	precision = scores['tp'] / (scores['tp'] + scores['fp'])
	print("Recall: ", recall)
	print("Precision: ", precision)
	print("F1: ", (2 * recall * precision) / (recall + precision))  # micro

	metrics_per_labels = {}
	for label in topics:
		recall = scores_per_label[label]["tp"] / (scores_per_label[label]["tp"] + scores_per_label[label]["fn"])
		precision = scores_per_label[label]["tp"] / (scores_per_label[label]["tp"] + scores_per_label[label]["fp"])
		f1 = (2 * recall * precision) / (recall + precision)
		metrics_per_labels[label] = {"recall": recall, "precision": precision, "f1": f1}

	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(pd.DataFrame(metrics_per_labels))
