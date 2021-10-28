import torch
from multi_label_training.src.model import BertForMultiLabelSequenceClassification
from multi_label_training.src.metrics import Metrics
from multi_label_training.src.report import Report
from multi_label_training.src.checkpoint import Checkpoint


class Trainer(object):

    def __init__(self,
                 config,
                 train_dataloader,
                 validation_dataloader):

        self.config = config
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = BertForMultiLabelSequenceClassification.from_pretrained(config["pretrained_model"],
                                                                             num_labels=config["n_labels"])
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.metrics = Metrics(config)
        self.report = Report(config, self.metrics)
        self.checkpoint = Checkpoint(config, self.model, self.optimizer)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            validation_loss, validation_micro_f1 = self.metrics.compute_loss_and_micro_f1('validation')

            self.checkpoint.maybe_save_checkpoint(epoch, validation_loss, validation_micro_f1)
            self.report.report_wandb(epoch, current_lr)
            self.metrics.reset()

    def forward_pass(self, data):
        ids = data['ids'].to(self.device)
        mask = data['mask'].to(self.device)
        targets = data['targets'].to(self.device)

        outputs = self.model(ids, mask, labels=targets)

        loss = outputs.loss
        logits = outputs.logits
        return loss, logits, targets

    def train_step(self, data):

        loss, logits, targets = self.forward_pass(data)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.metrics.update_metrics(torch.sigmoid(logits), targets, loss.item())

    @torch.no_grad()
    def validation_step(self, data):
        loss, logits, targets = self.forward_pass(data)
        self.metrics.update_metrics(torch.sigmoid(logits), targets, loss.item(), mode="validation")








