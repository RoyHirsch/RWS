"""Training and evaluation loops."""

import numpy as np
import time

import torch
import torch.nn as nn

import evaluation as evaluation
import losses 
import stand_alone.random_walk as rw


class LossesName:
    cross_entropy = 'cross_entrophy_loss'
    smoothness = 'smoothness_loss'
    confidence = 'confidence_loss'


class PseudoLabelMethodName:
    forward_backward = 'forward_backward'
    random_walk = 'random_walk'


class SmoothingMethodName:
    laplace = 'laplace'
    mse = 'mse'


class TrainMetricLogger():
    """Helper class for logging temporal segmentation metrics during trainign."""

    def __init__(self):
        self.losses = []
        self.timestamps_correct = 0.
        self.num_timestamps = 0
        self.correct = 0.
        self.total = 0.
        self.num_vids = 0
        self.additional_losses = {}

    def update(self,
               labels,
               preds_list,
               mask,
               timestamp_inds,
               loss,
               losses=None):
        """
        Gets a batch of predictions from temporal model and update running statistics.

        Parameters:
            labels: Labels tensor [batch_size, max_len].
            preds_list: List with length num_stages, each entry is: [batch_size, max_len, num_classes].
            mask: A mask tensor [batch_size, max_len].
            timestamp_inds: List of tensors, length is batch_size, each tensor contains timestapms indecies.
        """

        self.losses.append(loss)

        _, preds = torch.max(preds_list[-1].data, 1)
        self.correct += ((preds == labels).float() *
                         mask[:, 0, :].squeeze(1)).sum().item()
        self.total += torch.sum(mask[:, 0, :]).item()

        for i, timestamps in enumerate(timestamp_inds):
            self.num_vids += 1
            self.num_timestamps += float(len(timestamps))
            self.timestamps_correct += (
                labels[i,
                       timestamps] == preds[i,
                                            timestamps]).float().sum().item()

        if losses:
            for k, v in losses.items():
                if k not in self.additional_losses:
                    self.additional_losses[k] = []
                self.additional_losses[k].append(v)

    def calc(self):
        mets = {
            'loss': np.mean(self.losses),
            'timestamps_acc': self.timestamps_correct / self.num_timestamps,
            'acc': float(self.correct) / self.total
        }
        if len(self.additional_losses):
            additional_losses = {
                k: np.mean(v)
                for k, v in self.additional_losses.items()
            }
            mets.update(additional_losses)
        return mets


def run_train_epoch(train_data_loader, model, optimizer, epoch, config):
    model.train()
    cross_entropy = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
    mets = TrainMetricLogger()
    start = time.time()
    for video_names, featuers, labels, confidence_masks, masks, timestamps in train_data_loader:
        featuers, labels, masks, confidence_masks = featuers.to(
            config.device), labels.to(config.device), masks.to(
                config.device), confidence_masks.to(config.device)

        optimizer.zero_grad()
        final_representations, predictions = model(featuers, masks)

        losses_mapping = {
            k: 0.0
            for k in [
                LossesName.cross_entropy, LossesName.smoothness,
                LossesName.confidence
            ]
        }

        batch_labels = torch.ones_like(labels, dtype=torch.long) * (-100)
        batch_labels = batch_labels.to(config.device)

        ############################################################################################################
        # Warmup training with timestamps supervision
        ############################################################################################################
        if epoch < config.start_epochs:
            for i, (instance_timestamps,
                    instance_labels) in enumerate(zip(timestamps, labels)):
                batch_labels[
                    i,
                    instance_timestamps] = instance_labels[instance_timestamps]

        ############################################################################################################
        # Generate dense psaudo labels from timestamps
        ############################################################################################################
        else:
            pseudo_labels_acc = []
            for i, (instance_representations, instance_labels, t) in enumerate(
                    zip(final_representations, labels, timestamps)):

                if config.supervised_loss_name == PseudoLabelMethodName.forward_backward:
                    instance_representations, instance_labels = trim_instance(
                        instance_representations, instance_labels,
                        config.ignore_index)
                    preds = losses.get_forward_backward_dense_pseudo_labels(
                        instance_representations, instance_labels, t)

                elif config.supervised_loss_name == PseudoLabelMethodName.random_walk:
                    random_walk_params = {
                        'similarity_method': config.similarity_method,
                        'sharpening_method': config.sharpening_method,
                        'beta': config.beta,
                        'average_method': config.averaging_method,
                        'num_neighbors': config.num_neighbors,
                        'gamma': config.gamma,
                        'smooth': config.smooth
                    }
                    for i, (instance_representations, instance_labels,
                            t) in enumerate(
                                zip(final_representations, labels,
                                    timestamps)):  # TODO features !!!!!!!!
                        instance_representations, instance_labels = trim_instance(
                            instance_representations, instance_labels,
                            config.ignore_index)
                        preds = rw.get_random_walk_dense_pseudo_labels_from_timestamps(
                            instance_representations.T, instance_labels, t,
                            config.num_classes, random_walk_params)
                else:
                    raise ValueError('Invalid supervised loss name {}'.format(
                        config.supervised_loss_name))

                pseudo_labels_acc.append(
                    np.mean(instance_labels.numpy() == preds))
                batch_labels[i, :len(preds)] = torch.from_numpy(preds).to(
                    config.device)

        # Per-calculate the laplacians for the batch
        if config.smoothness_loss_name == SmoothingMethodName.laplace:
            affinities, laplacians = losses.calc_random_walk_dependencies_batch(
                final_representations, labels, config)

        ############################################################################################################
        # Calculate per-stage losses and summerize
        ############################################################################################################
        for stage_preds in predictions:
            losses_mapping[LossesName.cross_entropy] += cross_entropy(
                stage_preds.transpose(2, 1).contiguous().view(
                    -1, config.num_classes), batch_labels.view(-1))

            if config.smoothness_loss_name == SmoothingMethodName.mse:
                losses_mapping[
                    LossesName.
                    smoothness] += losses.calc_trunced_smoothing_mse_loss(
                        stage_preds)

            if config.smoothness_loss_name == SmoothingMethodName.laplace:
                losses_mapping[
                    LossesName.
                    smoothness] += losses.calc_trunced_smoothing_laplacian_loss(
                        laplacians, affinities, labels, stage_preds,
                        config.ignore_index)

            if config.confidence_weight > 0.:
                losses_mapping[LossesName.confidence] += losses.calc_confidence_loss(
                    stage_preds, confidence_masks)

        ############################################################################################################
        # Losses reduction
        ############################################################################################################
        loss = losses_mapping[LossesName.cross_entropy]
        detailed_losses = {
            LossesName.cross_entropy:
            losses_mapping[LossesName.cross_entropy].item()
        }

        if config.smoothness_weight > 0.:
            smoothness_loss = losses_mapping[
                LossesName.smoothness] * config.smoothness_weight
            loss += smoothness_loss
            detailed_losses[LossesName.smoothness] = smoothness_loss.item()

        if config.confidence_weight > 0.:
            confidence_loss = losses_mapping[
                LossesName.confidence] * config.confidence_weight
            loss += confidence_loss
            detailed_losses[LossesName.confidence] = confidence_loss.item()

        loss.backward()
        optimizer.step()

        mets.update(labels.detach().cpu(),
                    predictions.detach().cpu(),
                    masks.detach().cpu(), [t for t in timestamps], loss.item(),
                    detailed_losses)

    end = time.time()
    mets_dict = mets.calc()
    mets_dict['time'] = end - start
    if epoch >= config.start_epochs:
        mets_dict['pseudo_labels_acc'] = np.mean(pseudo_labels_acc)
    return mets_dict


def run_train_epoch_with_precalc_laplacian(train_data_loader, model, optimizer,
                                           epoch, video_name2lapacian, config):
    model.train()
    cross_entropy = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
    mets = TrainMetricLogger()
    start = time.time()
    for video_names, featuers, labels, masks, timestamps in train_data_loader:
        featuers, labels, masks = featuers.to(config.device), labels.to(
            config.device), masks.to(config.device)

        optimizer.zero_grad()
        _, predictions = model(featuers, masks)

        loss = 0.0
        if epoch < config.start_epochs:
            for stage_preds in predictions:
                loss += cross_entropy(stage_preds, labels)
        else:
            for stage_preds in predictions:
                loss += losses.calc_random_walk_loss_with_precalc_laplacian(
                    [
                        video_name2lapacian[video_name]
                        for video_name in video_names
                    ],
                    labels,
                    stage_preds.permute(0, 2, 1),
                    timestamps,
                    smoothness_weight=config.rw.smoothness_weight)
        mets.update(labels.detach().cpu(),
                    predictions.detach().cpu(),
                    masks.detach().cpu(), [t for t in timestamps], loss.item())
        loss.backward()
        optimizer.step()

    end = time.time()
    mets_dict = mets.calc()
    mets_dict['time'] = end - start
    return mets_dict


def run_eval(test_data_loader, model, config):
    model.eval()
    actions_dict_rev = {v: k for k, v in config.actions_dict.items()}
    met = evaluation.MetricLoger()
    with torch.no_grad():
        for video_names, featuers, labels, confidence_masks, masks, timestamps in test_data_loader:
            featuers, labels, masks, confidence_masks = featuers.to(
                config.device), labels.to(config.device), masks.to(
                    config.device), confidence_masks.to(config.device)
            _, predictions = model(featuers, masks)
            _, predicted = torch.max(predictions[-1].data, 1)
            labels = labels.detach().cpu().numpy()
            predicted = predicted.detach().cpu().numpy()
            for l, p in zip(labels, predicted):
                lim = np.where(l == config.ignore_index)[0]
                if len(lim):
                    lim = min(lim)
                else:
                    lim = len(l)
                met.update([actions_dict_rev[i] for i in l[:lim]],
                           [actions_dict_rev[i] for i in p[:lim]])
    return met.calc()


def trim_instance(features, labels, ignore_index):
    """Trims padded instance and detach."""

    padding_indexs = (labels == ignore_index).nonzero()
    lim = min(padding_indexs) if len(padding_indexs) else len(labels)
    features = features[:, :lim].detach().cpu().numpy()
    labels = labels[:lim].detach().cpu()
    return features, labels
