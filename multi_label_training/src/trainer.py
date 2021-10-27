import torch
from model import TransformerModel
from metrics import Metrics
from report import Report
from checkpoint import Checkpoint


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
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.metrics = Metrics(config)
        self.report = Report(config, self.metrics)
        self.checkpoint = Checkpoint(config, self.model, self.optimizer)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def loss_fn(self, outputs, targets):
        return torch.nn.BCELoss(reduction="sum")(outputs, targets) / targets.labels.size()[0]

    def train(self):

        for epoch in range(1, self.config["epochs"] + 1):

            train_progress_bar = self.report.reset_progress_bar(self.train_dataloader)
            self.model.to(self.device)
            self.model.train()

            for batch_idx, data in enumerate(train_progress_bar):
                self.train_step(data)
                if batch_idx % 50 == 0:
                    self.report.report_progress_bar(epoch, mode="train")

            validation_progress_bar = self.report.reset_progress_bar(self.validation_dataloader)
            self.model.eval()

            for batch_idx, data in enumerate(validation_progress_bar):
                self.validation_step(data)
                if batch_idx % 50 == 0:
                    self.report.report_progress_bar(epoch, mode="validation")

            self.scheduler.step(epoch)
            current_lr = self.scheduler.get_last_lr()
            validation_loss, validation_micro_f1 = self.metrics.compute_loss_and_micro_f1('validation')

            if validation_loss > self.metrics.best_validation_loss \
                    and validation_micro_f1 < self.metrics.best_validation_micro_f1:
                self.early_stopping_counter = self.early_stopping_counter + 1
            else:
                self.early_stopping_counter = 0

            self.checkpoint.maybe_save_checkpoint(epoch, validation_loss, validation_micro_f1)
            self.report.report_wandb(epoch, current_lr)
            self.metrics.reset()

    def train_step(self, data):
        ids = data['ids'].to(self.device)
        mask = data['mask'].to(self.device)
        targets = data['targets'].to(self.device)

        outputs = self.model(ids, mask)

        self.optimizer.zero_grad()
        loss = self.loss_fn(outputs, targets).item()
        loss.backward()
        self.optimizer.step()
        self.metrics.update_metrics(outputs, targets, loss)

    @torch.no_grad()
    def validation_step(self, data):
        ids = data['ids'].to(self.device)
        mask = data['mask'].to(self.device)
        targets = data['targets'].to(self.device)

        outputs = self.model(ids, mask)

        loss = self.loss_fn(outputs, targets).item()
        self.metrics.update_metrics(outputs, targets, loss, mode="validation")







