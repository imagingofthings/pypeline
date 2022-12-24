# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

r"""
Field synthesizers.

Let :math:`I_{k}(r)` be the :math:`k`-th energy level of a Bluebild estimate.
Given the compact description of energy levels :math:`D \in \mathbb{R}^{N}, V \in \mathbb{C}^{N \times N}`,
field synthesizers compute the value of :math:`I_{k}(r)` using the interpolation operator
ideally-matched to the instrument's sampling operator :math:`\Psi = \Phi W` :

.. math::

   I_{k}(r) = D_{k} \frac{|\Psi V_{k}|^{2}}{\langle V_{k}, G_{\Psi} V_{k} \rangle}.

Subclasses of :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.FieldSynthesizerBlock`
can be used to evaluate :math:`\{I_{k}(r), k = 1, \ldots, N\}`.

* If only interested in a single snapshot :math:`I_{k}(r, t)` or *very widely-spaced* snapshots
  :math:`\{I_{k}(r, t + q \Delta t), q = 0, \ldots, N_{t}\}`, use
  :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.spatial_domain.SpatialFieldSynthesizerBlock`.
* In all other cases, use :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.fourier_domain.FourierFieldSynthesizerBlock`.
"""

import pypeline.core as core


class FieldSynthesizerBlock(core.Block):
    """
    Top-level public interface for Bluebild field synthesizers.
    """

    def __init__(self):
        """

        """
        super().__init__()
        self.timer = None
        self.timer_tag = ""

    def __call__(self, *args, **kwargs):
        """
        Compute instantaneous field statistics.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        stat : :py:class:`~numpy.ndarray`
            Field statistics.
        """
        raise NotImplementedError

    def set_timer(self,t,tname = ""):
        self.timer = t
        self.timer_tag = tname

    def mark(self, tag):
      if self.timer: self.timer.start_time(tag)
    def unmark(self, tag):
      if self.timer: self.timer.end_time(tag)

    def synthesize(self, *args, **kwargs):
        """
        Compute field values from statistics.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        field : :py:class:`~numpy.ndarray`
            Field values.
        """
        raise NotImplementedError
