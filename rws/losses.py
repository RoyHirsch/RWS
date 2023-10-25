"""Different loss functions."""

from typing import Sequence, Optional, Mapping, Callable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stand_alone.random_walk import get_scores_sharpening_func, get_affinity_matrix, sparsify_affine_matrix


def get_forward_backward_dense_pseudo_labels(features, labels, timestamps):
    """Calculates the dense pseudo labels based on forward backward heuristic."""
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
        
    boundary_target = np.ones_like(labels) * (-100)
    boundary_target[:timestamps[0]] = labels[timestamps[0]]
    left_bound = [0]

    # Forward to find action boundaries
    for i in range(len(timestamps) - 1):
        start = timestamps[i]
        end = timestamps[i + 1] + 1
        left_score = torch.zeros(end - start - 1, dtype=torch.float)
        for t in range(start + 1, end):
            center_left = torch.mean(features[:, left_bound[-1]:t], dim=1)
            diff_left = features[:, start:t] - center_left.reshape(-1, 1)
            score_left = torch.mean(torch.norm(diff_left, dim=0))

            center_right = torch.mean(features[:, t:end], dim=1)
            diff_right = features[:, t:end] - center_right.reshape(-1, 1)
            score_right = torch.mean(torch.norm(diff_right, dim=0))

            left_score[t - start -
                       1] = ((t - start) * score_left +
                             (end - t) * score_right) / (end - start)

        cur_bound = torch.argmin(left_score) + start + 1
        left_bound.append(cur_bound.item())

    # Backward to find action boundaries
    right_bound = [len(labels)]
    for i in range(len(timestamps) - 1, 0, -1):
        start = timestamps[i - 1]
        end = timestamps[i] + 1
        right_score = torch.zeros(end - start - 1, dtype=torch.float)
        for t in range(end - 1, start, -1):
            center_left = torch.mean(features[:, start:t], dim=1)
            diff_left = features[:, start:t] - center_left.reshape(-1, 1)
            score_left = torch.mean(torch.norm(diff_left, dim=0))

            center_right = torch.mean(features[:, t:right_bound[-1]], dim=1)
            diff_right = features[:, t:end] - center_right.reshape(-1, 1)
            score_right = torch.mean(torch.norm(diff_right, dim=0))

            right_score[t - start -
                        1] = ((t - start) * score_left +
                              (end - t) * score_right) / (end - start)

        cur_bound = torch.argmin(right_score) + start + 1
        right_bound.append(cur_bound.item())

    # Average two action boundaries for same segment and generate pseudo labels
    left_bound = left_bound[1:]
    right_bound = right_bound[1:]
    num_bound = len(left_bound)
    for i in range(num_bound):
        temp_left = left_bound[i]
        temp_right = right_bound[num_bound - i - 1]
        middle_bound = int((temp_left + temp_right) / 2)
        boundary_target[timestamps[i]:middle_bound] = labels[timestamps[i]]
        boundary_target[middle_bound:timestamps[i + 1] +
                        1] = labels[timestamps[i + 1]]

    boundary_target[timestamps[-1]:] = labels[
        timestamps[-1]]  # frames after last single frame has same label
    return boundary_target


def calc_confidence_loss(logits, confidence_mask):
    """Calculates the confidence loss from:
    Temporal Action Segmentation from Timestamp Supervision."""

    confidence_mask = confidence_mask.permute(0, 2, 3, 1)
    batch_size = logits.size(0)
    pred = F.log_softmax(logits, dim=1)
    loss = 0
    for b in range(batch_size):
        num_frame = confidence_mask[b].shape[2]
        m_mask = confidence_mask[b].type(torch.float)
        left = pred[b, :, 1:] - pred[b, :, :-1]
        left = torch.clamp(left[:, :num_frame] * m_mask[0], min=0)
        left = torch.sum(left) / torch.sum(m_mask[0])
        loss += left

        right = (pred[b, :, :-1] - pred[b, :, 1:])
        right = torch.clamp(right[:, :num_frame] * m_mask[1], min=0)
        right = torch.sum(right) / torch.sum(m_mask[1])
        loss += right

    return loss


def calc_trunced_smoothing_laplacian_loss(laplacians, affinities, labels,
                                          preds, ignore_index):
    """Calculates the trunced laplacian smoothing loss."""

    total = 0.
    count = 0.
    for laplacian, affinity_matrix, instance_labels, instance_preds in zip(
            laplacians, affinities, labels, preds):
        padding_indexs = (instance_labels == ignore_index).nonzero()
        lim = min(padding_indexs) if len(padding_indexs) else len(
            instance_labels)
        instance_preds = instance_preds[:, :lim]
        for class_preds in F.log_softmax(instance_preds, dim=0):
            dists = torch.clamp(
                (class_preds[:-1] - class_preds[1:].detach())**2,
                min=0,
                max=16).type(torch.float64)

            right_weights = torch.diagonal(affinity_matrix, 1)
            right_weights = right_weights / right_weights.sum()
            left_weights = torch.diagonal(affinity_matrix, -1)
            left_weights = left_weights / left_weights.sum()
            total += 0.5 * (right_weights.dot(dists) + left_weights.dot(dists))
            count += 1
    return total / count


def calc_trunced_smoothing_mse_loss(stage_preds):
    return torch.mean(
        torch.clamp(nn.MSELoss(reduction='none')(
            F.log_softmax(stage_preds[:, :, 1:], dim=1),
            F.log_softmax(stage_preds.detach()[:, :, :-1], dim=1)),
                    min=0,
                    max=16))


def calc_random_walk_loss_with_precalc_laplacian(
        batch_laplacians: Sequence[torch.Tensor],
        batch_labels: torch.Tensor,
        batch_logits: torch.Tensor,
        timestamp_inds: Sequence[torch.Tensor],
        smoothness_weight: float = 0.1,
        padding_value: float = -100):
    """Calculates the random walk loss with constanct / precalc laplacisn."""

    loss = 0.0
    for laplacian, labels, logits, timestamps in zip(batch_laplacians,
                                                     batch_labels,
                                                     batch_logits,
                                                     timestamp_inds):
        laplacian = laplacian.to(logits.device)
        # trim padding, if any
        padding_indexs = (labels == padding_value).nonzero()
        lim = min(padding_indexs) if len(padding_indexs) else len(labels)
        logits = logits[:lim, :]
        labels = labels[:lim]

        # normalize the logits row-wise (each row corresponds to a frame's probabilites over the classes)
        preds = F.softmax(logits, dim=1)
        # iterate over all the classes and sum the smoothness loss
        for class_pred in preds.T:
            loss += smoothness_weight * class_pred @ laplacian @ class_pred

        # calculate the cross-entropy loss for the timestamps
        # TODO(royhirsch) define the sup. loss as callable
        loss += F.cross_entropy(logits[timestamps, :], labels[timestamps])

    return loss


def calc_random_walk_dependencies_batch(final_representations, labels, config):
    """Calculates dependencies for online random walk."""

    laplacians = []
    affinities = []
    for instance_representations, instance_labels in zip(
            final_representations, labels):
        padding_indexs = (instance_labels == config.ignore_index).nonzero()
        lim = min(padding_indexs) if len(padding_indexs) else len(
            instance_labels)
        instance_representations = instance_representations[:, :lim].detach(
        ).cpu().numpy()
        affinity = get_affinity_matrix(
            instance_representations.T,
            similarity_method=config.similarity_method,
            post_process_fn=get_scores_sharpening_func(
                config.sharpening_method, config.similarity_method,
                config.beta))
        affinity = sparsify_affine_matrix(
            affinity,
            num_neighbors=config.num_neighbors,
            average_method=config.averaging_method)
        affinity = torch.from_numpy(affinity).to(config.device).type(
            torch.float64)
        affinities.append(affinity)
        laplacian = torch.diag(affinity.sum(1)) - affinity
        laplacians.append(laplacian)
    return affinities, laplacians



    return torch.mean(
        torch.clamp(nn.MSELoss(reduction='none')(
            F.log_softmax(stage_preds[:, :, 1:], dim=1),
            F.log_softmax(stage_preds.detach()[:, :, :-1], dim=1)),
                    min=0,
                    max=16))
