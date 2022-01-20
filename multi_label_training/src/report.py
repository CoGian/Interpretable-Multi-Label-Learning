import os
import wandb

from tqdm import tqdm


class Report(object):
    def __init__(self, config, metrics):
        super(Report, self).__init__()
        self.config = config
        self.metrics = metrics
        self.progress_bar_writer = None

        self.wandb_writer = wandb
        project = os.environ.get(config['project'], config['logger_file'])
        self.wandb_writer.init(name=config['project'], project=project)

    def report_progress_bar(self, epoch, mode='train'):

        if self.config["token_classification"]:
            loss, micro_f1, micro_f1_per_input_id = self.metrics.compute_loss_and_micro_f1(mode)
            self.progress_bar_writer.set_description(
                'Epoch:{} - {}_loss {:.3f} | {}_micro_f1: {:.3f} | {}_micro_f1_per_input_id: {:.3f}'.format(
                    epoch,
                    mode,
                    loss,
                    mode,
                    micro_f1,
                    mode,
                    micro_f1_per_input_id)
            )
        else:
            loss, micro_f1 = self.metrics.compute_loss_and_micro_f1(mode)

            self.progress_bar_writer.set_description(
                'Epoch:{} - {}_loss {:.3f} | {}_micro_f1: {:.3f}'.format(
                    epoch,
                    mode,
                    loss,
                    mode,
                    micro_f1)
            )
            
        self.progress_bar_writer.update()

    def report_wandb(self, epoch, learning_rate):
        if self.wandb_writer is None:
            return
        
        if self.config["token_classification"]:
            train_loss, train_micro_f1, train_micro_precision, train_micro_recall,\
                train_micro_f1_per_input_id, train_micro_precision_per_input_id, train_micro_recall_per_input_id = \
                self.metrics.compute_epoch_metrics(mode='train')
            validation_loss, validation_micro_f1, validation_micro_precision, validation_micro_recall,\
                validation_micro_f1_per_input_id, validation_micro_precision_per_input_id, validation_micro_recall_per_input_id = \
                self.metrics.compute_epoch_metrics(mode='validation')
            test_loss, test_micro_f1, test_micro_precision, test_micro_recall, \
            test_micro_f1_per_input_id, test_micro_precision_per_input_id, test_micro_recall_per_input_id = \
                self.metrics.compute_epoch_metrics(mode='test')
        else:
            train_loss, train_micro_f1, train_micro_precision, train_micro_recall =\
                self.metrics.compute_epoch_metrics(mode='train')
            validation_loss, validation_micro_f1, validation_micro_precision, validation_micro_recall =\
                self.metrics.compute_epoch_metrics(mode='validation')
            test_loss, test_micro_f1, test_micro_precision, test_micro_recall = \
                self.metrics.compute_epoch_metrics(mode='test')

        log_dir = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_micro_f1": train_micro_f1,
            "train_micro_precision": train_micro_precision,
            "train_micro_recall": train_micro_recall,
            "validation_loss": validation_loss,
            "validation_micro_f1": validation_micro_f1,
            "validation_micro_precision": validation_micro_precision,
            "validation_micro_recall": validation_micro_recall,
            "test_loss": test_loss,
            "test_micro_f1": test_micro_f1,
            "test_micro_precision": test_micro_precision,
            "test_micro_recall": test_micro_recall,
        }

        if self.config["token_classification"]:
            log_dir["train_micro_f1_per_input_id"] = train_micro_f1_per_input_id
            log_dir["train_micro_precision_per_input_id"] = train_micro_precision_per_input_id
            log_dir["train_micro_recall_per_input_id"] = train_micro_recall_per_input_id
            log_dir["validation_micro_f1_per_input_id"] = validation_micro_f1_per_input_id
            log_dir["validation_micro_precision_per_input_id"] = validation_micro_precision_per_input_id
            log_dir["validation_micro_recall_per_input_id"] = validation_micro_recall_per_input_id
            log_dir["test_micro_f1_per_input_id"] = test_micro_f1_per_input_id
            log_dir["test_micro_precision_per_input_id"] = test_micro_precision_per_input_id
            log_dir["test_micro_recall_per_input_id"] = test_micro_recall_per_input_id

        self.wandb_writer.log(log_dir)

    def reset_progress_bar(self, dataloader):
        self.progress_bar_writer = tqdm(dataloader)
        return self.progress_bar_writer
