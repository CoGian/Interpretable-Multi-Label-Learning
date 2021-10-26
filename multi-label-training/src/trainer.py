import torch
from model import TransformerModel


class Trainer(object):

    def __init__(self,
                 config,
                 train_dataloader,
                 validation_dataloader):

        self.config = config
        self.early_stopping_counter = 0
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = TransformerModel(self.config)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'])
        self.steps_per_epoch = config['steps_per_epoch']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def loss_fn(self, outputs, targets):
        return torch.nn.BCELoss(reduction="sum")(outputs, targets) / targets.labels.size()[0]

    def train(self):

        for epoch in range(1, self.config["epochs"] + 1):

            self.model.to(self.device)
            self.model.train()

            for data in self.train_dataloader:
                self.train_step(data)

            for data in self.validation_dataloader:
                self.validation_step(data)

    def train_step(self, data):
        ids = data['ids'].to(self.device)
        mask = data['mask'].to(self.device)
        targets = data['targets'].to(self.device)

        outputs = self.model(ids, mask)

        self.optimizer.zero_grad()
        loss = self.loss_fn(outputs, targets).item()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def validation_step(self, data):
        ids = data['ids'].to(self.device)
        mask = data['mask'].to(self.device)
        targets = data['targets'].to(self.device)

        outputs = self.model(ids, mask)

        loss = self.loss_fn(outputs, targets).item()








