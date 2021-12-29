import argparse

import yaml
from multi_label_training.src.load_configurations import load_configs
from multi_label_training.src.dataset import CsvDataset, get_dataloader, JsonDataset
from multi_label_training.src.trainer import Trainer
from transformers import AutoTokenizer, set_seed

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--configs_path', '-cp', help='The path of the configuration file')
	args = parser.parse_args()
	configs_file = args.configs_path
	with open(configs_file, 'r') as config_file:
		configs = yaml.load(config_file, Loader=yaml.FullLoader)
	config = load_configs(configs)

	set_seed(44)
	tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"])

	if config["logger_file"] == "LitCovid":
		train_dataset = CsvDataset(
			config["train_file"],
			config["topics"],
			tokenizer,
			config["max_length"]
		)
		validation_dataset = CsvDataset(
			config["val_file"],
			config["topics"],
			tokenizer,
			config["max_length"]
		)
	elif config["logger_file"] == "HoC" or config["logger_file"] == "cei":
		train_dataset = JsonDataset(
			config["train_file"],
			config["topics"],
			tokenizer,
			config["max_length"],
			config["token_classification"]
		)
		validation_dataset = JsonDataset(
			config["val_file"],
			config["topics"],
			tokenizer,
			config["max_length"],
			config["token_classification"]
		)
	else:
		exit()

	train_dataloader = get_dataloader(train_dataset, config["batch_size"], True)
	validation_dataloader = get_dataloader(validation_dataset, config["batch_size"], False)

	trainer = Trainer(
		config=config,
		train_dataloader=train_dataloader,
		validation_dataloader=validation_dataloader)

	trainer.train()
