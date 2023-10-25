"""Componenets for stand-alone random walk for temporal action segmentation."""

from typing import Optional, Callable, Union, Sequence, Any
import functools
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy import sparse
from scipy.sparse import csgraph
import numpy as np


def get_affinity_matrix(frames: np.array,
                        similarity_method: str,
                        post_process_fn: Callable = lambda x: x):
    """Calculates the dense similarity matrix between the frames' reprentations.
  
  Attributse:
    frames: A numpy array of shape [num_frames, hidden_dim].
    similarity_method: The simillarity method name, supports: ['cosine', 'euclidean', 'norm_euclidean'].
    post_process_fn: Optional function for post processing.

  Returns:
    A scores matrix of shape [num_frames, num_frames]
  """

    if similarity_method == 'cosine':
        weights_matrix = cosine_similarity(frames, frames)
        # normalize to yield positive weights
        weights_matrix = (weights_matrix + 1.) / 2.

    elif similarity_method == 'euclidean':
        weights_matrix = euclidean_distances(frames, frames, squared=True)
        max_value = weights_matrix.max()
        # normalize and make sure there are no zeros
        weights_matrix = (weights_matrix / max_value) + 1e-6

    else:
        raise ValueError
    return post_process_fn(weights_matrix)


def nonezero_min(arr: np.ndarray):
    return arr[arr != 0].min()


def nonezero_mean(arr: np.ndarray):
    return arr[arr != 0].mean()


def sparsify_affine_matrix(weights_matrix: np.ndarray, num_neighbors: int,
                           average_method: str):
    """Sparsifies the affine matrix.

  Keep only scores of the current frame and nearest neighbors (t-1, t, t+1).
  Also includes a mechnism for averaging the weights of 'm' neighbors (from each side).

  Attributse:
   w_mat: The dense affine matrix of shape [num_frames, num_frames].
   num_neighbors: The number of neighbors to aggregate weights (from one-side).
   average_method: The method for averaging the m weight's values, supports: ['min', 'mean', 'max', 'none'].
   When 'none' is used, each node will have 2 * num_neighbors neighboring nodes.

  Returs:
   sparse_w_mat, same shape as w_mat
  """
    n = len(weights_matrix)
    inds = np.arange(n)
    rs = [
        np.array([
            weights_matrix[i, j] if j < n else 0.0
            for j in range(i + 1, i + num_neighbors + 1)
        ]) for i in range(n)
    ]
    ls = [
        np.array([
            weights_matrix[i, j] if j >= 0 else 0.0
            for j in range(i - num_neighbors, i)
        ]) for i in range(n)
    ]

    r = np.stack(rs, 1)
    l = np.flip(np.stack(ls, 1), 0)

    sparse_weights_matrix = np.zeros_like(weights_matrix) + np.identity(n)
    if average_method == 'mean':
        sparse_weights_matrix[inds[:-1], inds[:-1] + 1] = np.apply_along_axis(
            nonezero_mean, 0, r[:, :-1])
        sparse_weights_matrix[inds[:-1] + 1, inds[:-1]] = np.apply_along_axis(
            nonezero_mean, 0, l[:, 1:])

    elif average_method == 'min':
        sparse_weights_matrix[inds[:-1], inds[:-1] + 1] = np.apply_along_axis(
            nonezero_min, 0, r[:, :-1])
        sparse_weights_matrix[inds[:-1] + 1, inds[:-1]] = np.apply_along_axis(
            nonezero_min, 0, l[:, 1:])

    elif average_method == 'max':
        sparse_weights_matrix[inds[:-1], inds[:-1] + 1] = r.max(0)[:-1]
        sparse_weights_matrix[inds[:-1] + 1, inds[:-1]] = l.max(0)[1:]

    # dont do any averaging, define 2 * num_neighbors connections per node
    else:
        for i in range(1, num_neighbors + 1):
            sparse_weights_matrix[inds[:-i], inds[:-i] + i] = r[i - 1, :-i]
            sparse_weights_matrix[inds[:-i] + i, inds[:-i]] = l[i - 1, i:]

    return sparse_weights_matrix


def power(x, beta):
    return np.power(x, beta)


def exp(x, beta):
    return np.exp(-beta * x)


def get_scores_sharpening_func(sharpnening_method: str, similarity_method: str,
                               beta: float):
    if sharpnening_method == 'power' and similarity_method == 'cosine':
        return functools.partial(power, beta=beta)
    elif sharpnening_method == 'exp' and similarity_method == 'euclidean':
        return functools.partial(exp, beta=beta)
    # the affine matrix scores should be possitive
    elif sharpnening_method in ['none', 'no', None]:
        return lambda x: x
    else:
        raise ValueError(
            'Invalid combination of sharpening: {} and similarity {}'.format(
                sharpnening_method, similarity_method))


def solve(laplacian_matix: np.ndarray,
          prior: np.ndarray,
          gamma: Union[np.ndarray, float] = 1e-2):
    """Solves the array of linear equesions.

  Solves (L + gamma * I) x_s = gamma * z_j, where:
    L is laplacian matrix with shape [num_frames, num_frames]
    z_j is the prior vector with shape [num_frames]
    Gamma is a weight factor that controls the prior strenght, can be a vector or a scalar.
  This function solves this linear equesion for every phase/class.

  Attributse:
    laplacian_matix: The laplacian matrix with shape [num_frames, num_frames].
    prior: The prior matrix with shape [num_phases, num_frames].
    gamma: Weight factor that controls the prior strenght.

  Returns:
    The phase/class probability per frame, a matrix with shape:[num_phases, num_frames]
  """
    n = len(laplacian_matix)
    assert n == prior.shape[1]

    if isinstance(gamma, (int, float)):
        gamma_vector = np.full(shape=(n, ), fill_value=gamma)
    else:
        gamma_vector = gamma

    lap_sparse = sparse.csr_matrix(laplacian_matix)
    gamma_sparse = sparse.coo_matrix((gamma_vector, (range(n), range(n))))
    A_sparse = lap_sparse + gamma_sparse
    A_sparse = A_sparse.tocsc()
    solver = sparse.linalg.factorized(A_sparse.astype(np.double))
    X = np.array([solver(gamma_vector * label_prior) for label_prior in prior])
    return X


def predict(laplacian_matix: np.ndarray,
            prior: np.ndarray,
            gamma: Union[np.ndarray, int] = 1e-2):
    """Solves and predicts the array of linear equesions.
  
  Solves (L + gamma * I) x_s = gamma * z_j, where:
  L is laplacian matrix with shape [num_frames, num_frames]
  z_j is the prior vector with shape [num_frames]
  Gamma is a weight factor that controls the prior strenght, can be a vector or a scalar.
  This function solves this linear equesion for every phase/class and returns the class prediction per frame.

  Attributse:
    laplacian_matix: The laplacian matrix with shape [num_frames, num_frames].
    prior: The prior matrix with shape [num_phases, num_frames].
    gamma: Weight factor that controls the prior strenght.

  Returns:
    The phase/class prediction per frame, a vector:[num_frames]
  """

    X = solve(laplacian_matix, prior, gamma)
    preds = np.argmax(X, axis=0)
    return preds


def get_sparse_prior_from_timestamps(labels: np.ndarray,
                                     timestamp_indices: Union[Sequence[int],
                                                              np.ndarray],
                                     num_phases: int,
                                     value: float = 1.,
                                     smooth: Optional[int] = None):
    """Extracts sparse prior matrix based on timestamps.

  Given a list of timestamps indices, generate a sparse prior matrix with shape [num_phases, num_frames].
  This sparse matrix contains only the timestamps labels and the other entries are masked.
  The function also with smoothing the timestams labels.

  Attributse:
    labels: A list of ground true lables with shape [num_frames].
    timestamps: A list of timestamps indices.
    num_phases: The number of phases.
    value: The value for the timestamps locations.
    smooth: Optional smoothing factor.

  Returns:
    A sparse_prior matrix with shape [num_phases, num_frames]
  """

    num_frames = len(labels)
    sparse_prior = np.zeros((num_phases, num_frames))
    for j in timestamp_indices:
        i = labels[j]
        sparse_prior[i, j] = value
        if smooth:
            for m, n in enumerate(range(j - smooth, j)):
                if n >= 0:
                    sparse_prior[i, n] = (value / (smooth + 1) * (m + 1))
            for m, n in enumerate(range(j + smooth, j, -1)):
                if n < num_frames:
                    sparse_prior[i, n] = (value / (smooth + 1) * (m + 1))
    return sparse_prior


def predict_random_walk_with_prior(frames: np.ndarray, prior: np.ndarray,
                                   similarity_method: str,
                                   sharpening_method: str, beta: float,
                                   average_method: str, num_neighbors: int,
                                   gamma: Union[np.ndarray, float]):
    """Runs and predicts the random walk with prior.

  Attributse:
    frames: The video's frames with shape [num_frames, hidden_dim].
    prior: The prior matrix with shape [num_phases, num_frames].
    sharpening_method: Method name for the scores' sharpning.
    beta: Parameter for the sharpning mechanism.
    average_method: Scores averaging methode name.
    num_neighbors: The number of neighbors for scores averaging.
    gamma: The prior weight.

  Returns:
    The phase/class prediction per frame, a vector:[num_frames]

  """
    weights_matrix = get_affinity_matrix(
        frames,
        similarity_method=similarity_method,
        post_process_fn=get_scores_sharpening_func(sharpening_method,
                                                   similarity_method, beta))
    sparse_weights_matrix = sparsify_affine_matrix(
        weights_matrix,
        num_neighbors=num_neighbors,
        average_method=average_method)
    laplacian_matrix = csgraph.laplacian(sparse_weights_matrix, normed=False)
    preds = predict(laplacian_matrix, prior, gamma)
    return preds


DEFAULT_PARAMS = {
    'similarity_method': 'euclidean',
    'sharpening_method': 'exp',
    'beta': 30,
    'average_method': 'mean',
    'num_neighbors': 15,
    'gamma': 5e-3,
    'smooth': 10
}


def get_random_walk_dense_pseudo_labels_from_timestamps(
        frames, labels, timestamps, num_phases, params=DEFAULT_PARAMS):
    """Runs and predicts the random walk with timestamps prior.

  Attributse:
    frames: The video's frames with shape [num_frames, hidden_dim].
    labels: A list of ground true lables with shape [num_frames].
    timestamps: A list of timestamps indices.
    num_phases: The number of phases.
    params: A dictionary with hyper-paramaters.

  Returns:
    The phase/class prediction per frame, a vector:[num_frames]

  """
    sparse_prior = get_sparse_prior_from_timestamps(labels,
                                                    timestamps,
                                                    num_phases,
                                                    value=1.,
                                                    smooth=params['smooth'])
    return predict_random_walk_with_prior(
        frames=frames,
        prior=sparse_prior,
        sharpening_method=params['sharpening_method'],
        similarity_method=params['similarity_method'],
        beta=params['beta'],
        average_method=params['average_method'],
        num_neighbors=params['num_neighbors'],
        gamma=params['gamma'])
