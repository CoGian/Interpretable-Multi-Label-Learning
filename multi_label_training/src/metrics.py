import sys 
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(object):
	def __init__(self, config):
		self.config = config
		self.steps = dict(train=0, validation=0)
		self.micro_f1 = dict(train=0, validation=0)
		self.micro_precision = dict(train=0, validation=0)
		self.micro_recall = dict(train=0, validation=0)
		self.loss_metric = dict(train=0, validation=0)
		self.best_validation_loss = sys.float_info.max
		self.best_validation_micro_f1 = sys.float_info.min
		self.threshold = config['threshold']

	def update_metrics(
			self,
			outputs,
			targets,
			loss,
			mode='train'):
		"""
		Compute micro F1 for each batch and update the epoch's metrics
		:param outputs: The output of the model for this batch
		:param targets: The targets for each instance in this batch
		:param loss: The computed batch loss
		:param mode: Update train or val loss
		"""
		micro_f1, micro_precision, micro_recall = self.compute_batch_metrics(outputs, targets, mode)

		self.steps[mode] += 1
		self.loss_metric[mode] += loss
		self.micro_f1[mode] += micro_f1
		self.micro_precision[mode] += micro_precision
		self.micro_recall[mode] += micro_recall

	def compute_loss_and_micro_f1(self, mode='validation'):
		"""
		Compute loss and micro f1
		"""
		loss = self.loss_metric[mode] / self.steps[mode]
		micro_f1 = self.micro_f1[mode] / self.steps[mode]

		if mode == 'validation':
			self.best_validation_loss = min(self.best_validation_loss, loss)
			self.best_validation_micro_f1 = max(self.best_validation_micro_f1, micro_f1)

		return loss, micro_f1

	def compute_epoch_metrics(self, mode='train'):
		"""
		Calculate the loss and F1 metrics at the end of an epoch
		"""
		loss = self.loss_metric[mode] / self.steps[mode]
		micro_f1 = np.mean(self.micro_f1[mode]) / self.steps[mode]
		micro_precision = np.mean(self.micro_precision[mode]) / self.steps[mode]
		micro_recall = np.mean(self.micro_recall[mode]) / self.steps[mode]

		return loss, micro_f1, micro_precision, micro_recall

	def compute_batch_metrics(self, outputs, targets, mode='train'):
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

	def reset(self):
		"""
		Reset losses and metrics and the end of an epoch
		"""
		self.steps = dict(train=0, validation=0)
		self.micro_f1 = dict(train=0, validation=0)
		self.loss_metric = dict(train=0, validation=0)
		self.micro_recall = dict(train=0, validation=0)
		self.micro_precision = dict(train=0, validation=0)