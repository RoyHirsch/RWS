"""Main RWS (Random Walk Segmentation) training script."""

import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import logging
import argparse
import torch
from torch import optim

from model import MultiStageModel
from train import run_train_epoch_with_precalc_laplacian, run_eval, run_train_epoch
from config import get_config
from random_walk_utils import calc_video_name2lapacian
from data import get_dataloaders
import utils as utils

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-c',
    '--config',
    help='Path to config YAML file, if None loads default config',
    required=False,
    default=None)
parser.add_argument('-d',
                    '--data_path',
                    help='Optional data path',
                    required=False,
                    default=None)
parser.add_argument('-er',
                    '--exp_root',
                    help='Optional exp root',
                    required=False,
                    default=None)
parser.add_argument('-en',
                    '--exp_name',
                    help='Optional exp name',
                    required=False,
                    default=None)
args = parser.parse_args()


def run_experiment(config):

    utils.set_initial_random_seed(config.seed)
    utils.create_logger(log_dir=config.exp_dir, dump=config.dump_log)
    writer = utils.TensorBoardSummery(config.exp_dir)
    logging.info('Configuration:')
    logging.info(config)

    logging.info('#' * 50)
    logging.info('Start experiment : {}'.format(config.exp_name))
    logging.info('#' * 50)

    model = MultiStageModel(config.num_stages, config.num_layers,
                            config.num_f_maps, config.features_dim,
                            config.num_classes)
    model.to(config.device)
    logging.info('Init {} model with {} trainable parameters'.format(
        getattr(model, 'name', ''), utils.count_parameters(model)))
    model.train()

    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr,
                           weight_decay=config.weight_decay)

    if config.checkpoint:
        logging.info('Loading model checkpoint from: {}'.format(
            config.checkpoint))
        model = torch.load(config.checkpoint, map_location=config.device)
        model.to(config.device)
        model.train()

    train_data_loader, test_data_loader = get_dataloaders(config)
    logging.info(
        'Loaded train dataset with {} examples and test dataset with {} examples.'
        .format(len(train_data_loader.dataset), len(test_data_loader.dataset)))
    if config.i3d_based_laplacian:
        logging.info('Train with pre-calculated laplasian')
        video_name2laplacian = calc_video_name2lapacian(
            dataset=train_data_loader.dataset,
            sharpening_method=config.sharpening_method,
            beta=config.beta,
            averaging_method=config.averaging_method,
            num_neighbors=config.num_neighbors,
            device=config.device)
    else:
        logging.info('Train with learnable laplasian')

    logging.info(
        'Start training for dataset: {} split: {} with device: {}.\nTrain for {} epochs with {} warmup epochs.'
        .format(config.dataset, config.split, config.device, config.num_epochs,
                config.start_epochs))
    best_model_saver = utils.SaveBestModel(exp_dir=config.exp_dir,
                                           metric_to_monitor='loss',
                                           bigger_is_better=False,
                                           is_from_train=True)

    train_loss_list = []
    train_timestamps_acc_list = []
    train_labels_acc_list = []
    for epoch in range(config.num_epochs):
        if config.i3d_based_laplacian:
            train_epoch_mets = run_train_epoch_with_precalc_laplacian(
                train_data_loader, model, optimizer, epoch,
                video_name2laplacian, config)
            writer.add_dict(train_epoch_mets, epoch, 'train')
        else:
            train_epoch_mets = run_train_epoch(train_data_loader, model,
                                               optimizer, epoch, config)
            writer.add_dict(train_epoch_mets, epoch, 'train')
        logging.info(
            "[E{} | {:.0f}s | LR {:.5f}] Train || loss: {:.3f} | timestamps acc: {:.3f} | labels acc: {:.3f}"
            .format(epoch, train_epoch_mets['time'],
                    optimizer.param_groups[0]['lr'], train_epoch_mets['loss'],
                    train_epoch_mets['timestamps_acc'],
                    train_epoch_mets['acc']))

        if len([k for k in train_epoch_mets.keys() if 'loss' in k]) > 1:
            additonal_losses_str = ' | '.join([
                '{} : {:.3f}'.format(k.rstrip('_loss'), v)
                for k, v in train_epoch_mets.items() if 'loss' in k
            ])
            logging.info('Detailed losses : {}'.format(additonal_losses_str))

        train_loss_list.append(train_epoch_mets['loss'])
        train_timestamps_acc_list.append(train_epoch_mets['timestamps_acc'])
        train_labels_acc_list.append(train_epoch_mets['acc'])

        if config.eval_interval and epoch % config.eval_interval == 0 and epoch >= config.start_epochs:
            val_epoch_mets = run_eval(test_data_loader, model, config)
            writer.add_dict(val_epoch_mets, epoch, 'val')
            logging.info('[E{}] Val || '.format(epoch) + ' | '.join([
                '{} : {:.3f}'.format(k, v) for k, v in val_epoch_mets.items()
            ]))
            best_model_saver.update(train_epoch_mets, val_epoch_mets, epoch,
                                    model)

    best_model_saver.report_best()
    writer.add_hparams(config,
                       final_metric_dict=val_epoch_mets,
                       best_metric_dict=best_model_saver.val_best_metrics_dict)
    utils.save_model(config.exp_dir, epoch, model, optimizer=None)
    return val_epoch_mets, best_model_saver.val_best_metrics_dict


def main():
    config = get_config(args)
    _ = run_experiment(config)


if __name__ == '__main__':
    main()