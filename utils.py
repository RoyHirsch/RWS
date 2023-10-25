import logging
import os
import time
import numpy as np
import torch
import random
from datetime import timedelta
import json
from torch.utils.tensorboard import SummaryWriter


class TensorBoardSummery:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
    
    def add_scalar(self, metric_name, matric_value, epoch):
        self.writer.add_scalar(metric_name, matric_value, epoch)

    def add_hparams(self, config, final_metric_dict=None, best_metric_dict=None):
        config_dict = {k: v for k, v in dict(config).items() if type(v) in [int, str, float, torch.Tensor]}
        metric_dict = {'final_' + k: v for k, v in final_metric_dict.items()}
        metric_dict.update({'best_' + k: v for k, v in best_metric_dict.items()})
        self.writer.add_hparams(config_dict, metric_dict=metric_dict)
        text_string = ' \n '.join(['{} : {}'.format(k, str(v)) for k, v in dict(config).items()])
        self.writer.add_text(tag='hparams', text_string=text_string)

    def add_dict(self, metrics_dict, epoch, state=None):
        if state:
            for k, v in metrics_dict.items():
                self.writer.add_scalar('{}/{}'.format(state, k), v, epoch)
        else:
            for k, v in metrics_dict.items():
                self.writer.add_scalar(k, v, epoch)


def set_initial_random_seed(random_seed):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(log_dir, dump=True, file_name='launcher.log'):
    filepath = os.path.join(log_dir, file_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create logger
    log_formatter = LogFormatter()

    if dump:
        # create file handler and set level to info
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to info
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if dump:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info('Created main log at ' + str(filepath))
    return logger


def save_model(exp_dir, epoch, model, file_name='', optimizer=None, verbose=True):
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    torch.save(model, os.path.join(exp_dir, file_name + 'epoch-' + str(epoch + 1) + '.pt'))
    if optimizer:
        torch.save(optimizer, os.path.join(exp_dir,  file_name + 'opt-epoch-' + str(epoch + 1) + '.pt'))
    if verbose:
        logging.info('Saved model {}'.format(os.path.join(exp_dir,  file_name + 'epoch-' + str(epoch + 1) + '.pt')))


class SaveBestModel:

    def __init__(self, exp_dir, metric_to_monitor, is_from_train=True, bigger_is_better=True, file_name='', verbose=True):
        self.metric_to_monitor = metric_to_monitor
        self.is_from_train = is_from_train
        self.bigger_is_better = bigger_is_better
        self.verbose = verbose
        string = file_name + '_best_model.pt' if len(file_name) else 'best_model.pt'
        self.full_name = os.path.join(exp_dir, string)
        self.best_value = 0.0 if bigger_is_better else 1e10
        self.val_best_metrics_dict = None

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

    def _save_model(self, model, epoch):
        torch.save(model, self.full_name)
        if self.verbose:
            logging.info('[E{}] Save new best model'.format(epoch))
    
    def _log_best_model(self, current_value, val_metrics_dict, epoch, model):
        self._save_model(model, epoch)
        self.best_value = current_value
        val_metrics_dict.update({'epoch': epoch})
        self.val_best_metrics_dict = val_metrics_dict

    def update(self, train_metrics_dict, val_metrics_dict, epoch, model):
        if self.is_from_train:
            current_value = train_metrics_dict[self.metric_to_monitor]
        else:
            current_value = val_metrics_dict[self.metric_to_monitor]
        if self.bigger_is_better:
            if current_value >= self.best_value:
                self._log_best_model(current_value, val_metrics_dict, epoch, model)
        else:
            if current_value <= self.best_value:
                self._log_best_model(current_value, val_metrics_dict, epoch, model)

    def report_best(self):
        epoch = self.val_best_metrics_dict.pop('epoch')
        logging.info('Best model was in epoch: {}'.format(epoch))
        metrics_str = ' | '.join(['{} : {:.3f}'.format(k, v) for k, v in self.val_best_metrics_dict.items()])
        logging.info('Best model metrics : {}'.format(metrics_str))

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
