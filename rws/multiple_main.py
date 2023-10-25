"""Utils for setting and running multiple experiments."""

import os

import json
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import get_and_modify_config
from rws_main import run_experiment


def copy_and_modify(base_params, params_to_modify):
    params = copy.deepcopy(base_params)
    for k, v in params_to_modify.items():
        params[k] = v
    return params


def run_experiments_over_all_folds(base_config):
    if base_config.dataset == '50salads':
        splits = list(range(1, 6))
    else:
        splits = list(range(1, 5))

    splits_final_metric_dicts = []
    splits_best_metric_dicts = []

    for split in splits:
        config = copy_and_modify(base_config,
                                {'exp_dir': os.path.join(base_config.exp_root, base_config.exp_name, 'split{}'.format(split)),
                                 'exp_name' : '{}_split{}'.format(base_config.exp_name, split),
                                 'vid_list_file' : os.path.join(base_config.data_root_dir, base_config.dataset, 'splits', 'train.split{}.bundle'.format(split)),
                                 'vid_list_file_tst': os.path.join(base_config.data_root_dir, base_config.dataset, 'splits', 'test.split{}.bundle'.format(split)),
                                 'split': str(split)})
        final_metric_dict, best_metric_dict = run_experiment(config)
        splits_final_metric_dicts.append(final_metric_dict)        
        splits_best_metric_dicts.append(best_metric_dict)        
        
    # Report mean metrics
    mean_final_metrics = {}
    writer = SummaryWriter(os.path.join(base_config.exp_root, base_config.exp_name))
    for metric_name in splits_final_metric_dicts[0].keys():
        matric_value = np.mean([d[metric_name] for d in splits_final_metric_dicts])
        matric_std = np.std([d[metric_name] for d in splits_final_metric_dicts])
        mean_final_metrics[metric_name] = {'mean': matric_value, 'std': matric_std}
    writer.add_scalar('final_mean_' + metric_name, matric_value, 0)

    mean_best_metrics = {}
    for metric_name in splits_best_metric_dicts[0].keys():
        matric_value = np.mean([d[metric_name] for d in splits_best_metric_dicts])
        matric_std = np.std([d[metric_name] for d in splits_best_metric_dicts])
        mean_best_metrics[metric_name] = {'mean': matric_value, 'std': matric_std}
    writer.add_scalar('best_mean_' + metric_name, matric_value, 0)

    with open(os.path.join(base_config.exp_root, base_config.exp_name, 'mean_mets.json'), 'w') as f:
        json.dump({'final': mean_final_metrics, 'best': mean_best_metrics},
                    f, indent=4)
    return mean_final_metrics, mean_best_metrics


def losses_scan():
    base_config = {
        'supervised_loss_name': 'random_walk',
        'smoothness_loss_name': 'mse',
        'smoothness_weight': 0.15,
        'confidence_weight': 0.075,
        'num_epochs': 50,
        'start_epochs': 30,
        'dataset': 'breakfast',
        'split': '1',
        'gpu_num': 3,
        'exp_name': ''}
    exps = []
    for confidence_weight in [0.05, 0.075, 0.1]:
        for smoothness_weight in [0.075, 0.15]:
            exps.append(copy_and_modify(base_config,
                                        {'smoothness_weight': smoothness_weight,
                                        'confidence_weight': confidence_weight,
                                        'exp_name': 'breakfast_1_conf={:.4f}_lap={:.4f}_beta=30'.format(
                                            confidence_weight, smoothness_weight)}))

    return exps
if __name__ == '__main__':
    losses_scan()
