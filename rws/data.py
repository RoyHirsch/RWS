"""Datasets and Dataloaders for the tested three datasets."""

from typing import Any, Tuple, Union
import os
import torch
import numpy as np
import random
import functools

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class I3DRepresentationsDataset(Dataset):
    """Loads and processes I3D features of videos.""" 

    def __init__(self, actions_dict, dataset_name, gt_path, features_path, video_names_path, sample_rate):
        self.actions_dict = actions_dict
        self.num_classes = len(self.actions_dict)
        self.dataset_name = dataset_name
        self.gt_path = gt_path
        self.features_path = features_path
        self.video_names_path = video_names_path
        self.sample_rate = sample_rate

        with open(video_names_path, 'r') as f:
            self.video_names_list = f.read().split('\n')[:-1]
            
        self.video_name2timestamps = np.load(os.path.join(gt_path, dataset_name + '_annotation_all.npy'), allow_pickle=True).item()
        self.video_name2timestamps = {k: torch.tensor(v) for k, v in self.video_name2timestamps.items()}

        self.video_name2labels = {}
        for video_name in self.video_names_list:
            with open(os.path.join(self.gt_path, video_name), 'r') as f:
                content = f.read().split('\n')[:-1]
            classes = np.zeros(len(content))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            classes = classes[::self.sample_rate]
            self.video_name2labels[video_name] = torch.tensor(classes, dtype=torch.long)

        self.video_name2confidence_mask = self.generate_confidence_masks()
        
    def generate_confidence_masks(self):
        video_name2confidence_mask = {}
        for video_name in self.video_names_list:
            labels = self.video_name2labels[video_name]
            timestamps = self.video_name2timestamps[video_name]
            num_frames = labels.shape[0]

            left_mask = torch.zeros([self.num_classes, num_frames - 1])
            right_mask = torch.zeros([self.num_classes, num_frames - 1])
            for j in range(len(timestamps) - 1):
                left_mask[int(labels[timestamps[j]]), timestamps[j]:timestamps[j + 1]] = 1
                right_mask[int(labels[timestamps[j + 1]]), timestamps[j]:timestamps[j + 1]] = 1

            video_name2confidence_mask[video_name] = torch.stack([left_mask.T, right_mask.T], 1)
        return video_name2confidence_mask

    def __len__(self):
        return len(self.video_names_list)
    
    def __getitem__(self, ind):
        video_name = self.video_names_list[ind]

        features = torch.tensor(np.load(os.path.join(self.features_path, video_name.split('.')[0] + '.npy'))[:, ::self.sample_rate], dtype=torch.float32)
        labels = self.video_name2labels[video_name]
        timestamps = self.video_name2timestamps[video_name]
        onfidence_mask = self.video_name2confidence_mask[video_name]
        return video_name, features, labels, onfidence_mask, timestamps


def collate_and_pad_fn(batch: Any,
                       num_classes: int,
                       padding_values: Tuple[Union[int, float]]=(0.0, -100)):
    """Stacks and pads a batch of items with varing lengths.

    Attributes:
        num_classes: The number of classes.
        padding_values: A tuple for the padding values for features and labels.

    Returns:
        features: [batch_size, hidden_dim, max_len]
        labels: [batch_size, max_len]
        mask: A boolean mask, [batch_size, num_classes, max_len]
        vids_names: list of video names
        timestamp_inds: list of batch_size tensors with the timestamps inds
    """
    video_names = []
    featuers = []
    labels = []
    confidence_masks = []
    timestamps = []
    masks = []
    for item in batch:
        n, f, l, c, t = item
        video_names.append(n)
        featuers.append(f.T)
        labels.append(l)
        confidence_masks.append(c)
        timestamps.append(t)
        masks.append(torch.ones(len(l), num_classes))
    featuers = torch.nn.utils.rnn.pad_sequence(featuers,
                                               batch_first=True,
                                               padding_value=padding_values[0]).permute(0, 2, 1)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=padding_values[1])
    confidence_masks = torch.nn.utils.rnn.pad_sequence(confidence_masks,
                                                       batch_first=True,
                                                       padding_value=0.0)
    masks = torch.nn.utils.rnn.pad_sequence(masks,
                                            batch_first=True,
                                            padding_value=0.0).permute(0, 2, 1)
    return video_names, featuers, labels, confidence_masks, masks, timestamps
    

def get_dataloaders(config):
    train_dataset = I3DRepresentationsDataset(
                                  actions_dict=config.actions_dict,
                                  dataset_name=config.dataset,
                                  gt_path=config.gt_path,
                                  features_path=config.features_path,
                                  video_names_path=config.train_videos_list,
                                  sample_rate=config.sample_rate)

    test_dataset = I3DRepresentationsDataset(
                                 actions_dict=config.actions_dict,
                                 dataset_name=config.dataset,
                                 gt_path=config.gt_path,
                                 features_path=config.features_path,
                                 video_names_path=config.test_videos_list,
                                 sample_rate=config.sample_rate)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   collate_fn=functools.partial(collate_and_pad_fn,
                                   num_classes=config.num_classes),
                                   num_workers=config.num_workers,
                                   pin_memory=True)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  collate_fn=functools.partial(collate_and_pad_fn,
                                  num_classes=config.num_classes),
                                  num_workers=config.num_workers,
                                  pin_memory=True)
    return train_data_loader, test_data_loader


if __name__ == '__main__':
    from config import get_default_config

    config = get_default_config()
    dataset = I3DRepresentationsDataset(
                            actions_dict=config.actions_dict,
                            dataset_name=config.dataset,
                            gt_path=config.gt_path,
                            features_path=config.features_path,
                            video_names_path=config.vid_list_file,
                            sample_rate=config.sample_rate)
    item = dataset.__getitem__(0)

    data_loader = DataLoader(dataset,
                             batch_size=8,
                             shuffle=True,
                             collate_fn=functools.partial(collate_and_pad_fn,
                                                          num_classes=19),
                             num_workers=4,
                             pin_memory=True)
    batch = next(iter(data_loader))
