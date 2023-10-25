"""Pytorch compatible functions for random walk."""

from typing import Optional, Mapping, Callable
import functools
from scipy.stats import norm

import torch
import torch.nn.functional as F

from data import I3DRepresentationsDataset


class AggregationOperator():
    """Handles the aggreagetion of num neighbors similarity scores."""


    def __init__(self, method, num_neighbors):
        self.method = method
        self.num_neighbors = num_neighbors
        
        if method == 'linear_weighted_mean':
            self.w = torch.arange(1, num_neighbors + 1, dtype=torch.float32)
            s = self.w.sum()
            self.w /= s
        elif method == 'norm_weighted_mean':
            self.w = torch.tensor(norm(loc=0, scale=num_neighbors).pdf(range(num_neighbors)))
        else:
            pass

    def calc(self, x, right=True):
        if not len(x):
            return torch.tensor((0.0))
        elif self.method == 'mean':
            return x.mean()
        elif self.method == 'linear_weighted_mean':
            w = self.w if right else torch.flip(self.w, (0,))
            if len(x) < self.num_neighbors: w = w[:len(x)]
            return (w * x).sum()
        elif self.method == 'norm_weighted_mean':
            w = self.w if right else torch.flip(self.w, (0,))
            if len(x) < self.num_neighbors: w = w[:len(x)]
            return (w * x).sum()
        elif self.method == 'sum':
            return x.sum()
        elif self.method == 'min':
            return x.min()
        elif self.method == 'max':
            return x.max()
        else:
            raise ValueError


def power(x, beta):
  return torch.pow(x, beta)


def exp(x, beta):
  return torch.exp(-beta * x)


def get_scores_sharpening_func(sharpnening_method: str,
                               similarity_method: str,
                               beta: float):
  if sharpnening_method == 'power' and similarity_method == 'cosine':
    return functools.partial(power, beta=beta)
  elif sharpnening_method == 'exp' and similarity_method == 'euclidean':
    return functools.partial(exp, beta=beta)  
  elif sharpnening_method in ['none', 'no', None]:
    return lambda x: x
  else:
    raise ValueError('Invalid combination of sharpening: {} and similarity {}'.foramt(sharpnening_method, similarity_method))

  
def calc_similarity_matrix(frames: torch.Tensor,
                           similarity_method: str='cosine',
                           post_process_func: Callable = lambda x: x) -> torch.Tensor:
    """Calculates the cosine similarity between the frames' representations

    Attributes:
        frames: The frames representations, [max_len, hidden_dim]
        post_process_func: A function for processing the cosine scores (sharpning for example).
    Returns:
        Cosine similarity matrix [num_frames, num_frames]    
    """
    if similarity_method == 'cosine':
        frames_norm = F.normalize(frames, p=2, dim=1)
        similarity_matrix = torch.mm(frames_norm, frames_norm.transpose(0, 1))
        # the cosine similarity matrix is between [-1, 1]
        # we wish to fix it before applying the post processing to be between [0, 1]
        similarity_matrix = (similarity_matrix + 1.) / 2.

    elif similarity_method == 'euclidean':
        similarity_matrix = torch.mm(frames, frames.transpose(0, 1))
        max_value = similarity_matrix.max()
        similarity_matrix = (similarity_matrix / max_value) + 1e-6
    else:
        raise ValueError

    return post_process_func(similarity_matrix)


def calc_laplacian(adjacency_matrix: torch.Tensor,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    """Calcs laplacian given the adjacency matrix.
    """
    dagree_matrix = torch.diag(adjacency_matrix.sum(1))
    laplacian = dagree_matrix - adjacency_matrix
    if device: laplacian = laplacian.to(device)
    return laplacian


def calc_video_name2lapacian(dataset: I3DRepresentationsDataset,
                             sharpening_method: str = 'power',
                             similarity_method: str = 'cosine',
                             beta: float = 5,
                             averaging_method: str = 'mean',
                             num_neighbors: int = 20, 
                             device: torch.device = torch.device('cpu')) -> Mapping[str, torch.Tensor]:
    """Returns a video name to laplacian mapping.
    The laplacian calculation is based on pre-calculated features (in a torch Dataset).

    Attributse:
        dataset: Dataset of videos to evaluate.
        sharpening_method: Method name for the scores' sharpning.
        similarity_method: Method name for the scores' similarity.
        beta: Parameter for the sharpning mechanism.
        average_method: Scores averaging methode name.
        num_neighbors: The number of neighbors for scores averaging.
    """
    post_process_func = get_scores_sharpening_func(sharpening_method, similarity_method, beta)
    video_name2laplacian = {}
    for item in dataset:
        video_name, frames, _, _ = item
        similarity_matrix = calc_similarity_matrix(frames.T, post_process_func)

        # sparsify the similarity matrix (keep only the 2 main off-diagonal) (the 1NN scores from each side)
        if num_neighbors == 1:
            sparse_adjacency_matrix = torch.diag(torch.diagonal(similarity_matrix, 1), 1) + \
                                      torch.diag(torch.diagonal(similarity_matrix, -1), -1)
        else:
            # sparsify the similarity matrix (keep only the 2 main off-diagonal) (the aggregated kNN scores from each side)
            aggregator = AggregationOperator(averaging_method, num_neighbors)
            rs = []
            ls = []
            n = len(similarity_matrix)
            for i in range(n):
                rs.append(aggregator.calc(similarity_matrix[i, i + 1 : min(n, i + num_neighbors + 1)], True))
                ls.append(aggregator.calc(similarity_matrix[i, max(0, i - num_neighbors) : i], False))
            
            sparse_adjacency_matrix = torch.diag(torch.nan_to_num(torch.tensor(rs[1:]), 0.0), 1) + \
                                      torch.diag(torch.nan_to_num(torch.tensor(ls[:-1]), 0.0), -1)
        # construct the laplacian matrix
        video_name2laplacian[video_name] = calc_laplacian(sparse_adjacency_matrix).to(device)
    return video_name2laplacian
