# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

r"""
High-level Bluebild interfaces.

Let :math:`I_{k}(r,t)` denote the :math:`k`-th energy-level obtained at time :math:`t`.
Subclasses of :py:class:`~pypeline.phased_array.bluebild.imager.IntegratingMultiFieldSynthesizerBlock` do 3 things:

* integrate snapshot images to obtain integrated images spanning many observation periods: :math:`I_{k}(r) = \sum_{q=1}^{N_{t}} I_{k}(r, t_{q})`;
* aggregate energy levels;
* re-weight energy levels to output both a least-squares estimate and a standardized estimate of the (integrated) field.

Integrated images can then be directly output in viewable form by calling :py:meth:`~pypeline.phased_array.bluebild.imager.IntegratingMultiFieldSynthesizerBlock.as_image`.
"""

import imot_tools.util.argcheck as chk
import imot_tools.util.array as array
import numpy as np

import pypeline.core as core


class IntegratingMultiFieldSynthesizerBlock(core.Block):
    """
    Top-level public interface of Bluebild multi-field synthesizers.
    """

    def __init__(self):
        """

        """
        super().__init__()
        self._statistics = None

    def _update(self, stat):
        if self._statistics is None:
            self._statistics = stat
        else:
            self._statistics += stat

    def __call__(self, *args, **kwargs):
        """
        Compute integrated field statistics for least-squares and standardized estimates.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        stat : :py:class:`~numpy.ndarray`
            Integrated field statistics.
        """
        raise NotImplementedError

    def as_image(self):
        """
        Transform integrated statistics to viewable image.

        Returns
        -------
        std : :py:class:`~imot_tools.io.s2image.Image`
            (N_level, ...) standardized energy-levels.

        lsq : :py:class:`~imot_tools.io.s2image.Image`
            (N_level, ...) least-squares energy-levels.
        """
        raise NotImplementedError


@chk.check(dict(x=chk.is_array_like, idx=chk.has_integers, N=chk.is_integer, axis=chk.is_integer))
def cluster_layers(x, idx, N, axis):
    """
    Additive tensor compression along an axis.

    Parameters
    ----------
    x : array-like
        (..., K, ...) array.
    idx : array-like(int)
        (K,) cluster indices.
    N : int
        Total number of levels along compression axis.
    axis : int
        Dimension along which to compress.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        (..., N, ...) array
    """
    x = np.array(x, copy=False)
    idx = np.array(idx, copy=False)

    y_shape = list(x.shape)
    y_shape[axis] = N
    y = np.zeros(y_shape, dtype=x.dtype)

    for x_id, y_id in enumerate(idx):
        y[array.index(y, axis, y_id)] += x[array.index(x, axis, x_id)]
    return y
