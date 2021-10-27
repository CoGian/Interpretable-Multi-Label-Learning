import torch
from transformers import AutoModel


class TransformerModel(torch.nn.Module):
	def __init__(self, config):
		super(TransformerModel, self).__init__()
		self.l1 = AutoModel.from_pretrained(config["pretrained_model"])
		self.dropout = torch.nn.Dropout(config["dropout"])
		self.linear = torch.nn.Linear(self.l1.config.hidden_size, config['n_labels'])

	def forward(self, ids, mask):
		_, output_1 = self.l1(ids, attention_mask=mask)
		output_2 = self.dropout(output_1)
		output_3 = self.linear(output_2)
		output = torch.sigmoid(output_3)
		return output


