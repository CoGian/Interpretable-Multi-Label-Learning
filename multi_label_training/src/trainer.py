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
        if config["token_classification"]:
            self.model = BertForMultiLabelSequenceClassification.from_pretrained(config["pretrained_model"],
                                                                                 num_labels=config["n_labels"],
                                                                                 multi_task=True)
        else:
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
            
            if self.config["token_classification"]:
                validation_loss, validation_micro_f1, validation_micro_f1_per_input_id = \
                    self.metrics.compute_loss_and_micro_f1('validation')
            else:
                validation_loss, validation_micro_f1 = self.metrics.compute_loss_and_micro_f1('validation')

            self.checkpoint.maybe_save_checkpoint(epoch, validation_loss, validation_micro_f1)
            self.report.report_wandb(epoch, current_lr)
            self.metrics.reset()

    def forward_pass(self, data):
        ids = data['ids'].to(self.device)
        mask = data['mask'].to(self.device)
        targets = data['targets'].to(self.device)

        if self.config["token_classification"]:
            targets_per_input_id = data['targets_per_input_id'].to(self.device)
            outputs, outputs_per_input_id = self.model(
                ids,
                mask,
                labels=targets,
                targets_per_input_id=targets_per_input_id)
            loss = outputs.loss
            logits = outputs.logits

            loss_per_input_id = outputs_per_input_id[0]
            logits_per_input_id = outputs_per_input_id[1]

            return loss, logits, targets, loss_per_input_id, logits_per_input_id, targets_per_input_id
        else:
            outputs = self.model(ids, mask, labels=targets)

        loss = outputs.loss
        logits = outputs.logits
        return loss, logits, targets

    def train_step(self, data):

        if self.config["token_classification"]:
            loss, logits, targets, loss_per_input_id, logits_per_input_id, targets_per_input_id = self.forward_pass(data)

            total_loss = torch.add(loss, loss_per_input_id)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.metrics.update_metrics(
                torch.sigmoid(logits),
                targets,
                total_loss.item(),
                torch.sigmoid(logits_per_input_id),
                targets_per_input_id,
                data['mask'])
        else:
            loss, logits, targets = self.forward_pass(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.metrics.update_metrics(torch.sigmoid(logits), targets, loss.item())

    @torch.no_grad()
    def validation_step(self, data):
        if self.config["token_classification"]:
            loss, logits, targets, loss_per_input_id, logits_per_input_id, targets_per_input_id = self.forward_pass(data)
            total_loss = torch.add(loss, loss_per_input_id)
            self.metrics.update_metrics(
                torch.sigmoid(logits),
                targets,
                total_loss.item(),
                torch.sigmoid(logits_per_input_id),
                targets_per_input_id,
                data['mask'],
                mode="validation")
        else:
            loss, logits, targets = self.forward_pass(data)
            self.metrics.update_metrics(torch.sigmoid(logits), targets, loss.item(), mode="validation")








