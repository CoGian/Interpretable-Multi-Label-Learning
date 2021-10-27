import os
import wandb

from tqdm import tqdm


class Report(object):
    def __init__(self, config, metrics):
        super(Report, self).__init__()
        self.metrics = metrics
        self.progress_bar_writer = None

        self.wandb_writer = wandb
        project = os.environ.get(config['project'], config['logger_file'])
        self.wandb_writer.init(name=config['project'], project=project)

    def report_progress_bar(self, epoch, mode='train'):

        loss, micro_f1 = self.metrics.compute_loss_and_micro_f1(mode)

        self.progress_bar_writer.set_description(
            'Epoch:{} - {}_loss {:.3f} | {}_micro_f1: {:.3f}'.format(
                epoch,
                mode,
                loss,
                mode,
                micro_f1,
                mode)
        )

        self.progress_bar_writer.update()

    def report_wandb(self, epoch, learning_rate):
        if self.wandb_writer is None:
            return

        train_loss, train_micro_f1, train_micro_precision, train_micro_recall =\
            self.metrics.compute_epoch_metrics(mode='train')
        validation_loss, validation_micro_f1, validation_micro_precision, validation_micro_recall =\
            self.metrics.compute_epoch_metrics(mode='validation')

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
        }

        self.wandb_writer.log(log_dir)

    def reset_progress_bar(self, dataloader):
        self.progress_bar_writer = tqdm(dataloader)
        return self.progress_bar_writer
