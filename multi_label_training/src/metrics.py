import sys 
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(object):
	def __init__(self, config):
		self.config = config
		self.steps = dict(
			train=0, validation=0, test=0,
			train_per_input_id=0, validation_per_input_id=0, test_per_input_id=0)
		self.micro_f1 = dict(
			train=0, validation=0, test=0,
			train_per_input_id=0, validation_per_input_id=0, test_per_input_id=0)
		self.micro_precision = dict(
			train=0, validation=0, test=0,
			train_per_input_id=0, validation_per_input_id=0, test_per_input_id=0)
		self.micro_recall = dict(
			train=0, validation=0, test=0,
			train_per_input_id=0, validation_per_input_id=0, test_per_input_id=0)
		self.loss_metric = dict(train=0, validation=0, test=0)
		self.best_validation_loss = sys.float_info.max
		self.best_validation_micro_f1 = sys.float_info.min
		self.best_validation_micro_f1_per_input_id = sys.float_info.min
		self.threshold = config['threshold']

	def update_metrics(
			self,
			outputs,
			targets,
			loss,
			outputs_per_input_id=None,
			targets_per_input_id=None,
			attention_mask=None,
			mode='train'):
		"""
		Compute micro F1 for each batch and update the epoch's metrics
		:param outputs: The output of the model for this batch
		:param targets: The targets for each instance in this batch
		:param loss: The computed batch loss
		:param mode: Update train or val loss
		"""
		micro_f1, micro_precision, micro_recall = self.compute_batch_metrics(outputs, targets)
		self.steps[mode] += 1
		self.loss_metric[mode] += loss
		self.micro_f1[mode] += micro_f1
		self.micro_precision[mode] += micro_precision
		self.micro_recall[mode] += micro_recall

		if self.config["token_classification"]:
			micro_f1_per_input_id, micro_precision_per_input_id, micro_recall_per_input_id = \
				self.compute_batch_metrics_per_input_id(
					outputs_per_input_id,
					targets_per_input_id,
					attention_mask)

			self.micro_f1[mode+"_per_input_id"] += micro_f1_per_input_id
			self.micro_precision[mode+"_per_input_id"] += micro_precision_per_input_id
			self.micro_recall[mode+"_per_input_id"] += micro_recall_per_input_id

	def compute_loss_and_micro_f1(self, mode='validation'):
		"""
		Compute loss and micro f1
		"""
		loss = self.loss_metric[mode] / self.steps[mode]
		micro_f1 = self.micro_f1[mode] / self.steps[mode]
		
		if self.config["token_classification"]:
			micro_f1_per_input_id = self.micro_f1[mode+"_per_input_id"] / self.steps[mode]

		if mode == 'validation':
			self.best_validation_loss = min(self.best_validation_loss, loss)
			self.best_validation_micro_f1 = max(self.best_validation_micro_f1, micro_f1)
			if self.config["token_classification"]:
				self.best_validation_micro_f1_per_input_id = max(
					self.best_validation_micro_f1_per_input_id, micro_f1_per_input_id)

		if self.config["token_classification"]:
			return loss, micro_f1, micro_f1_per_input_id
		
		return loss, micro_f1

	def compute_epoch_metrics(self, mode='train'):
		"""
		Calculate the loss and F1 metrics at the end of an epoch
		"""
		loss = self.loss_metric[mode] / self.steps[mode]
		micro_f1 = np.mean(self.micro_f1[mode]) / self.steps[mode]
		micro_precision = np.mean(self.micro_precision[mode]) / self.steps[mode]
		micro_recall = np.mean(self.micro_recall[mode]) / self.steps[mode]

		if self.config["token_classification"]:
			micro_f1_per_input_id = np.mean(self.micro_f1[mode+"_per_input_id"]) / self.steps[mode]
			micro_precision_per_input_id = np.mean(self.micro_precision[mode+"_per_input_id"]) / self.steps[mode]
			micro_recall_per_input_id = np.mean(self.micro_recall[mode+"_per_input_id"]) / self.steps[mode]
			
			return loss, micro_f1, micro_precision, micro_recall, \
				   micro_f1_per_input_id, micro_precision_per_input_id, micro_recall_per_input_id

		return loss, micro_f1, micro_precision, micro_recall

	def compute_batch_metrics(self, outputs, targets):
		"""
		compute the micro F1 for each batch
		:param outputs: The output of the model for this batch
		:param targets: The targets for each instance in this batch
		:param mode: train or val
		"""

		outputs = outputs.cpu().data.numpy()
		numpy_targets = targets.cpu().data.numpy()
		predictions = np.where(outputs > self.threshold, 1, 0)
		micro_f1 = f1_score(y_true=numpy_targets, y_pred=predictions, average='micro')
		micro_precision = precision_score(y_true=numpy_targets, y_pred=predictions, average='micro')
		micro_recall = recall_score(y_true=numpy_targets, y_pred=predictions, average='micro')
		return micro_f1, micro_precision, micro_recall

	def compute_batch_metrics_per_input_id(self, outputs, targets, attention_masks):
		"""
		compute the micro F1 for each batch
		:param outputs: The output of the model for this batch
		:param targets: The targets for each instance in this batch
		:param mode: train or val
		"""

		outputs = outputs.cpu().data.numpy()
		numpy_targets = targets.cpu().data.numpy()
		predictions = np.where(outputs > self.threshold, 1, 0)

		tp = 0
		fp = 0
		fn = 0

		for target, prediction, mask in zip(numpy_targets, predictions, attention_masks):
			for token_target, token_prediction, token_mask in zip(target, prediction, mask):
				if token_mask == 1:
					for i in range(len(token_prediction)):
						if token_target[i] == token_prediction[i] == 1:
							tp += 1
						elif token_prediction[i] == 1 and token_target[i] != token_prediction[i]:
							fp += 1
						elif token_prediction[i] == 0 and token_target[i] != token_prediction[i]:
							fn += 1

		try:
			micro_precision = tp / (tp+fp)
		except ZeroDivisionError:
			micro_precision = 0.0

		try:
			micro_recall = tp / (tp+fn)
		except ZeroDivisionError:
			micro_recall = 0.0

		try:
			micro_f1 = (2 * micro_recall * micro_precision) / (micro_recall + micro_precision)
		except ZeroDivisionError:
			micro_f1 = 0.0

		# print(micro_f1, micro_precision, micro_recall)
		return micro_f1, micro_precision, micro_recall

	def reset(self):
		"""
		Reset losses and metrics at the end of an epoch
		"""
		self.steps = dict(
			train=0, validation=0, test=0,
			train_per_input_id=0, validation_per_input_id=0, test_per_input_id=0)
		self.micro_f1 = dict(
			train=0, validation=0, test=0,
			train_per_input_id=0, validation_per_input_id=0, test_per_input_id=0)
		self.micro_precision = dict(
			train=0, validation=0, test=0,
			train_per_input_id=0, validation_per_input_id=0, test_per_input_id=0)
		self.micro_recall = dict(
			train=0, validation=0, test=0,
			train_per_input_id=0, validation_per_input_id=0, test_per_input_id=0)
		self.loss_metric = dict(train=0, validation=0, test=0)
