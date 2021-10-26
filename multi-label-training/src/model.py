import torch
import yaml
from transformers import AutoModel
from load_configurations import load_configs


class TransformerModel(torch.nn.Module):
	def __init__(self, configs):
		super(TransformerModel, self).__init__()
		self.l1 = AutoModel.from_pretrained(configs["pretrained_model"])
		self.dropout = torch.nn.Dropout(configs["dropout"])
		self.linear = torch.nn.Linear(self.l1.config.hidden_size, configs['n_labels'])

	def forward(self, ids, mask):
		_, output_1 = self.l1(ids, attention_mask=mask)
		output_2 = self.dropout(output_1)
		output_3 = self.linear(output_2)
		output = torch.sigmoid(output_3)
		return output


if __name__ == '__main__':
	with open('../configs/LitCovid_configs.yml', 'r') as config_file:
		configs = yaml.load(config_file, Loader=yaml.FullLoader)
	configs = load_configs(configs)
	model = TransformerModel(configs)
	print(model)
