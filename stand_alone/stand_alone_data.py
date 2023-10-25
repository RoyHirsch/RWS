"""Data loading for stand-alone feature."""

from typing import Mapping
import os
import numpy as np


def load_dataset(dataset_name: str = '50salads',
                 split: str = '1',
                 data_root_dir: str = '/mnt/disks/tempseg/data/data'):

    train_files_list = os.path.join(data_root_dir, dataset_name, 'splits',
                                    'train.split{}.bundle'.format(split))
    val_files_list = os.path.join(data_root_dir, dataset_name, 'splits',
                                  'test.split{}.bundle'.format(split))
    features_path = os.path.join(data_root_dir, dataset_name, 'features')
    labels_path = os.path.join(data_root_dir, dataset_name, 'groundTruth')
    mapping_file = os.path.join(data_root_dir, dataset_name, 'mapping.txt')
    sample_rate = 2 if dataset_name == '50salads' else 1

    with open(mapping_file, 'r') as f:
        actions = f.read().split('\n')[:-1]
    actions_dict = {}
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    metadata = {'actions_dict': actions_dict, 'num_classes': len(actions_dict)}
    train_data = load_dataset_fold(dataset_name, train_files_list,
                                   features_path, labels_path, actions_dict,
                                   sample_rate)
    val_data = load_dataset_fold(dataset_name, val_files_list, features_path,
                                 labels_path, actions_dict, sample_rate)
    return train_data, val_data, metadata


def load_dataset_fold(dataset_name: str, video_names_path: str,
                      features_path: str, labels_path: str,
                      actions_dict: Mapping[str, int], sample_rate: int):

    with open(video_names_path, 'r') as f:
        video_names_list = f.read().split('\n')[:-1]

    video_name2timestamps = np.load(os.path.join(
        labels_path, dataset_name + '_annotation_all.npy'),
                                    allow_pickle=True).item()
    video_name2timestamps = {
        k: np.asarray(v, dtype=int)
        for k, v in video_name2timestamps.items()
    }
    video_name2features = {}
    video_name2labels = {}
    for video_name in video_names_list:
        with open(os.path.join(labels_path, video_name), 'r') as f:
            content = f.read().split('\n')[:-1]
        labels = np.zeros(len(content), dtype=int)
        for i in range(len(labels)):
            labels[i] = actions_dict[content[i]]
        labels = labels[::sample_rate]
        video_name2labels[video_name] = labels

        features = np.load(
            os.path.join(features_path,
                         video_name.split('.')[0] + '.npy'))
        video_name2features[video_name] = features[:, ::sample_rate]

    data = []
    for video_name in video_names_list:
        data.append({
            'features': video_name2features[video_name].T,
            'labels': video_name2labels[video_name],
            'timestamps': video_name2timestamps[video_name],
            'video_name': video_name
        })
    return data