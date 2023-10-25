"""A simple script for running stand-alone random walk for temporal segmentation."""

import sys
import os
sys.path.append(os.getcwd())
                
import numpy as np
import argparse
from tqdm import tqdm

from stand_alone_data import load_dataset
from random_walk import get_random_walk_dense_pseudo_labels_from_timestamps
from evaluation import MetricLoger


parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-dataset_name',
    '--dataset_name',
    help='Dataset name: [gtea, breakfast, 50salads].',
    required=False,
    default='gtea')
parser.add_argument(
    '-limit',
    '--limit',
    help='Optional number to limit the number of examples.',
    required=False,
    default=None)

args = parser.parse_args()


def get_random_walk_params(dataset_name):
    if dataset_name == 'gtea':
        return {'similarity_method': 'euclidean',
                'sharpening_method': 'exp',
                'beta': 30,
                'average_method': 'min',
                'num_neighbors': 15,
                'gamma': 0.001,
                'smooth': 10}
    elif dataset_name == '50salads':
        return {'similarity_method': 'euclidean',
                'sharpening_method': 'exp',
                'beta': 30, 'average_method': 'min',
                'num_neighbors': 15,
                'gamma': 0.001,
                'smooth': 10}
    elif dataset_name == 'breakfast':
        return {'similarity_method': 'euclidean',
                'sharpening_method': 'exp',
                'beta': 30, 'average_method':
                'min', 'num_neighbors': 15,
                'gamma': 0.001,
                'smooth': 20}
    else:
        raise ValueError


def main():
    train_data, val_data, metadata = load_dataset(args.dataset_name)
    print('Load {} dataset with {} train and {} val examples'.format(
        args.dataset_name, len(train_data), len(val_data)))
    data = train_data + val_data
    random_walk_params = get_random_walk_params(args.dataset_name)

    mets_logger = MetricLoger()
    for i, item in enumerate(tqdm(data)):
        preds = get_random_walk_dense_pseudo_labels_from_timestamps(
            item['features'], item['labels'], item['timestamps'],
            metadata['num_classes'], random_walk_params)
        mets_logger.update(item['labels'], preds)
        if args.limit and i == args.limit:
            break
    print(mets_logger.calc())
    


if __name__ == '__main__':
    main()