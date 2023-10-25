from typing import Optional, Mapping, Callable, Union, Sequence,  Tuple
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt


ColorType = Tuple[float, float, float, float]
DEFAULT_COLORMAP_NAME = 'jet'

ArrayLike = Union[Sequence[Union[float, int]], np.ndarray]
IntArrayLike = Union[Sequence[int], np.ndarray]


def get_colors_list(n: int) -> Sequence[ColorType]:
  """Generate list of n colors (RGBA format)."""
  cmap = matplotlib.cm.get_cmap(DEFAULT_COLORMAP_NAME)
  return [cmap(i) for i in range(0, 256, 256 // n)]


def plot_labels_preds_timestamps(labels: ArrayLike,
                                 preds: ArrayLike,
                                 timestamps_indices: Optional[IntArrayLike] = np.array([]),
                                 phase2color: Optional[Mapping[int, ColorType]] = None):
  """Plots labels and preds bar with optional plotting of timestaps.

  Attributse:
    labels: An array of label, shape [num_frames].
    preds: An array of preds, shape [num_frames].
    timestamps_indices: Optional array of timestamps indices.
  """

  n = len(preds)
  x = np.arange(n)
  ss = [500] * n
  if not phase2color:
    phases = np.unique(np.concatenate([np.unique(np.asarray(preds)),
                                      np.unique(np.asarray(labels))]))
    phase2color = {p: c for p, c in zip(phases, get_colors_list(len(phases)))}
  if timestamps_indices.any():
    timestamps_labels = [labels[i] for i in timestamps_indices]
    plt.scatter(timestamps_indices, [1.1] * len(timestamps_indices), c=timestamps_labels, edgecolors="black", cmap='jet')
  plt.scatter(x, [1.2] * n, s=ss, marker="|", c=[phase2color[p] for p in preds], cmap='jet')
  plt.scatter(x, [1] * n, s=ss, marker="|", c=[phase2color[l] for l in labels], cmap='jet')

  if timestamps_indices.any():
    plt.annotate("Seeds",(1, 1.12))
  plt.annotate("GT",(1, 1.02))
  plt.annotate("Preds",(1, 1.22))

  plt.ylim([0.9, 1.3])
  plt.yticks([])
  plt.show()



def plot(mets_dict, title='', is_grid=True, is_legend=True):
    for k, v in mets_dict.items():
        plt.plot(v, label=str(k))
    plt.grid(is_grid)
    if is_legend:
        plt.legend()
    plt.title(title)
    plt.show()