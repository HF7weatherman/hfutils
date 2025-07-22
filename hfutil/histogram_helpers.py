import numpy as np

# ------------------------------------------------------------------------------
# Functions for 2D histogram analysis
# ------------------------------------------------------------------------------
def get_cond_hist2d(
        counts_2d_comp: np.ndarray,
        norm_bins: np.ndarray,
        norm_axis: str
        ) -> np.ndarray:
    """
    Get conditional 2d histogram from compound 2d histogram.

    Parameters
    ----------
    counts_2d_comp : ndarray
        The 2D array representing the histogram counts.
    norm_bins : ndarray
        The bin edges along the normalization axis.
    norm_axis : {'x', 'y'}
        The axis along which to normalize ('x' or 'y').

    Returns
    -------
    counts_2d_cond_x : ndarray
        The 2D array of conditional histogram counts normalized along the specified axis.
    """
    if norm_axis == 'x':
        counts_2d_cond = counts_2d_comp / np.expand_dims(np.diff(norm_bins), axis=0) / \
            counts_2d_comp.sum(axis=1, keepdims=True)
    elif norm_axis == 'y':
        counts_2d_cond = counts_2d_comp / np.expand_dims(np.diff(norm_bins), axis=1) / \
            counts_2d_comp.sum(axis=0, keepdims=True)
    return counts_2d_cond


def get_bin_centers(bin_edges: np.ndarray) -> np.ndarray:
    """
    Calculate the centers of bins given their edges.

    Parameters
    ----------
    bin_edges : np.ndarray
        Array of bin edges.

    Returns
    -------
    np.ndarray
        Array of bin centers.
    """
    return (bin_edges[:-1] + bin_edges[1:]) / 2