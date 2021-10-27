import yaml
from src.load_configurations import load_configs
from src.dataset import LitCovidDataset, get_dataloader
from src.trainer import Trainer
from transformers import AutoTokenizer

if __name__ == '__main__':
	with open('./configs/LitCovid_configs.yml', 'r') as config_file:
		configs = yaml.load(config_file, Loader=yaml.FullLoader)
	config = load_configs(configs)

	tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"])
	train_dataset = LitCovidDataset(
		config["train_file"],
		config["topics"],
		tokenizer,
		config["max_length"]
	)
	validation_dataset = LitCovidDataset(
		config["val_file"],
		config["topics"],
		tokenizer,
		config["max_length"]
	)

	train_dataloader = get_dataloader(train_dataset, config["batch_size"], True)
	validation_dataloader = get_dataloader(validation_dataset, config["batch_size"], False)

	trainer = Trainer(
		config=config,
		train_dataloader=train_dataloader,
		validation_dataloader=validation_dataloader)

	trainer.train()
