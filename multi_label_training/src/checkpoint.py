import os
import sys
import torch


class Checkpoint(object):

	def __init__(self, config, model, optimizer):
		self.config = config
		self.checkpoint_dir = os.path.join(config['checkpoint_folder'], config['checkpoint_subfolder'])
		self.optimizer = optimizer
		self.model = model
		self.checkpoint_validation_micro_f1 = sys.float_info.min
		self.checkpoint_validation_micro_f1_per_input_id = sys.float_info.min
		self.checkpoint_validation_loss = sys.float_info.max
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

	def maybe_save_checkpoint(
			self,
			epoch,
			validation_loss,
			validation_micro_f1,
			validation_micro_f1_per_input_id=sys.float_info.min):
		if self.checkpoint_validation_micro_f1 < validation_micro_f1\
				or self.checkpoint_validation_micro_f1_per_input_id < validation_micro_f1_per_input_id:
			if self.checkpoint_dir is None:
				return

			self.checkpoint_validation_loss = validation_loss
			self.checkpoint_validation_micro_f1 = validation_micro_f1

			if not os.path.exists(os.path.abspath(self.checkpoint_dir)):
				os.makedirs(os.path.abspath(self.checkpoint_dir))

			checkpoint_name = f'model-epoch={epoch}-validation_loss={validation_loss}-validation_micro_f1={validation_micro_f1}'
			checkpoint_to_save_path = os.path.join(os.path.abspath(self.checkpoint_dir), checkpoint_name)
			self.model.save_pretrained(checkpoint_to_save_path)
