# #############################################################################
# spatial_domain.py
# =================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
High-level Bluebild interfaces that work in Fourier Series domain.
"""

import imot_tools.util.argcheck as chk
import numpy as np
import scipy.sparse as sparse

import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as psd
import pypeline.phased_array.bluebild.imager as bim
import imot_tools.io.s2image as image
import pypeline.util.array as array
import imot_tools.math.sphere.transform as transform


class Fourier_IMFS_Block(bim.IntegratingMultiFieldSynthesizerBlock):
    """
    Multi-field synthesizer based on PeriodicSynthesis.

    Examples
    --------
    Assume we are imaging a portion of the Bootes field with LOFAR's 24 core stations.

    The short script below shows how to use :py:class:`~pypeline.phased_array.bluebild.imager.fourier_domain.Fourier_IMFS_Block` to form continuous integrated energy level estimates.

    .. testsetup::

       import numpy as np
       import astropy.units as u
       import astropy.time as atime
       import astropy.coordinates as coord
       import scipy.constants as constants
       from tqdm import tqdm as ProgressBar
       from pypeline.phased_array.bluebild.data_processor import IntensityFieldDataProcessorBlock
       from pypeline.phased_array.bluebild.imager.fourier_domain import Fourier_IMFS_Block
       from pypeline.phased_array.instrument import LofarBlock
       from pypeline.phased_array.beamforming import MatchedBeamformerBlock
       from pypeline.phased_array.bluebild.gram import GramBlock
       from pypeline.phased_array.data_gen.source import from_tgss_catalog
       from pypeline.phased_array.data_gen.statistics import VisibilityGeneratorBlock
       from imot_tools.math.sphere.grid import equal_angle
       from imot_tools.math.sphere.transform import pol2cart

       np.random.seed(0)

    .. doctest::

       ### Experiment setup ================================================
       # Observation
       >>> obs_start = atime.Time(56879.54171302732, scale='utc', format='mjd')
       >>> field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
       >>> field_of_view = np.deg2rad(5)
       >>> frequency = 145e6
       >>> wl = constants.speed_of_light / frequency

       # instrument
       >>> N_station = 24
       >>> dev = LofarBlock(N_station)
       >>> mb = MatchedBeamformerBlock([(_, _, field_center) for _ in range(N_station)])
       >>> gram = GramBlock()

       # Visibility generation
       >>> sky_model=from_tgss_catalog(field_center, field_of_view, N_src=10)
       >>> vis = VisibilityGeneratorBlock(sky_model,
       ...                                T=8,
       ...                                fs=196000,
       ...                                SNR=np.inf)

       ### Energy-level imaging ============================================
       # Kernel parameters
       >>> t_img = obs_start + np.arange(200) * 8 * u.s  # fine-grained snapshots
       >>> obs_start, obs_end = t_img[0], t_img[-1]
       >>> R = dev.icrs2bfsf_rot(obs_start, obs_end)
       >>> N_FS = dev.bfsf_kernel_bandwidth(wl, obs_start, obs_end)
       >>> T_kernel = np.deg2rad(10)

       # Pixel grid: make sure to generate it in BFSF coordinates by applying R.
       >>> _, _, px_colat, px_lon = equal_angle(N=dev.nyquist_rate(wl),
       ...                                      direction=np.dot(R, field_center.transform_to('icrs').cartesian.xyz.value),
       ...                                      FoV=field_of_view)

       >>> I_dp = IntensityFieldDataProcessorBlock(N_eig=7,  # assumed obtained from IntensityFieldParameterEstimator.infer_parameters()
       ...                                         cluster_centroids=[124.927,  65.09 ,  38.589,  23.256])
       >>> I_mfs = Fourier_IMFS_Block(wl, px_colat, px_lon, N_FS, T_kernel, R, N_level=4)
       >>> for t in ProgressBar(t_img):
       ...     XYZ = dev(t)
       ...     W = mb(XYZ, wl)
       ...     S = vis(XYZ, W, wl)
       ...     G = gram(XYZ, W, wl)
       ...
       ...     D, V, c_idx = I_dp(S, G)
       ...
       ...     # (2, N_eig, N_height, N_FS+Q) energy levels [integrated, clustered) (compact descriptor, not the same thing as [D, V]).
       ...     field_stat = I_mfs(D, V, XYZ.data, W.data, c_idx)

       >>> I_std, I_lsq = I_mfs.as_image()

    The standardized and least-squares images can then be viewed side-by-side:

    .. doctest::

       from imot_tools.io.s2image import Image
       import matplotlib.pyplot as plt

       # Transform grid to ICRS coordinates before plotting.
       px_grid = np.tensordot(R.T, pol2cart(1, px_colat, px_lon), axes=1)

       fig, ax = plt.subplots(ncols=2)
       I_std.draw(index=slice(None),  # Collapse all energy levels
                  catalog=sky_model.xyz.T,
                  data_kwargs=dict(cmap='cubehelix'),
                  catalog_kwargs=dict(s=600),
                  ax=ax[0])
       ax[0].set_title('Standardized Estimate')
       I_lsq.draw(index=slice(None),  # Collapse all energy levels
                  catalog=sky_model.xyz.T,
                  data_kwargs=dict(cmap='cubehelix'),
                  catalog_kwargs=dict(s=600),
                  ax=ax[1])
       ax[1].set_title('Least-Squares Estimate')
       fig.show()

    .. image:: _img/bluebild_FourierIMFSBlock_integrate_example.png
    """

    @chk.check(
        dict(
            wl=chk.is_real,
            grid_colat=chk.has_reals,
            grid_lon=chk.has_reals,
            N_FS=chk.is_odd,
            T=chk.is_real,
            R=chk.require_all(chk.has_shape([3, 3]), chk.has_reals),
            N_level=chk.is_integer,
            precision=chk.is_integer,
        )
    )
    def __init__(self, wl, grid_colat, grid_lon, N_FS, T, R, N_level, precision=64):
        r"""
        Parameters
        ----------
        wl : float
            Wavelength [m] of observations.
        grid_colat : :py:class:`~numpy.ndarray`
            (N_height, 1) BFSF polar angles [rad].
        grid_lon : :py:class:`~numpy.ndarray`
            (1, N_width) equi-spaced BFSF azimuthal angles [rad].
        N_FS : int
            :math:`2\pi`-periodic kernel bandwidth. (odd-valued)
        T : float
            Kernel periodicity [rad] to use for imaging.
        R : array-like(float)
            (3, 3) ICRS -> BFSF rotation matrix.
        N_level : int
            Number of clustered energy-levels to output.
        precision : int
            Numerical accuracy of floating-point operations.

            Must be 32 or 64.

        Notes
        -----
        * `grid_colat` and `grid_lon` should be generated using :py:func:`~imot_tools.math.sphere.grid.equal_angle`.
        * `N_FS` can be optimally chosen by calling :py:meth:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock.bfsf_kernel_bandwidth`.
        * `R` can be obtained by calling :py:meth:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock.icrs2bfsf_rot`.
        """
        super().__init__()

        if precision == 32:
            self._fp = np.float32
            self._cp = np.complex64
        elif precision == 64:
            self._fp = np.float64
            self._cp = np.complex128
        else:
            raise ValueError("Parameter[precision] must be 32 or 64.")

        if N_level <= 0:
            raise ValueError("Parameter[N_level] must be positive.")
        self._N_level = N_level

        self._synthesizer = psd.FourierFieldSynthesizerBlock(
            wl, grid_colat, grid_lon, N_FS, T, R, precision
        )

    @chk.check(
        dict(
            D=chk.has_reals,
            V=chk.has_complex,
            XYZ=chk.has_reals,
            W=chk.is_instance(np.ndarray, sparse.csr_matrix, sparse.csc_matrix),
            cluster_idx=chk.has_integers,
        )
    )
    def __call__(self, D, V, XYZ, W, cluster_idx):
        """
        Compute (clustered) integrated field statistics for least-squares and standardized estimates.

        Parameters
        ----------
        D : :py:class:`~numpy.ndarray`
            (N_eig,) positive eigenvalues.
        V : :py:class:`~numpy.ndarray`
            (N_beam, N_eig) complex-valued eigenvectors.
        XYZ : :py:class:`~numpy.ndarary`
            (N_antenna, 3) Cartesian instrument geometry.

            `XYZ` must be given in ICRS.
        W : :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.csr_matrix` or :py:class:`~scipy.sparse.csc_matrix`
            (N_antenna, N_beam) synthesis beamweights.
        cluster_idx : :py:class:`~numpy.ndarray`
            (N_eig,) cluster indices of each eigenpair.

        Returns
        -------
        stat : :py:class:`~numpy.ndarray`
            (2, N_level, N_height, N_FS + Q) field statistics.
        """
        D = D.astype(self._fp, copy=False)

        stat_std = self._synthesizer(V, XYZ, W)
        stat_lsq = stat_std * D.reshape(-1, 1, 1)

        stat = np.stack([stat_std, stat_lsq], axis=0)
        stat = bim.cluster_layers(stat, cluster_idx, N=self._N_level, axis=1)

        self._update(stat)
        return stat

    def as_image(self):
        """
        Transform integrated statistics to viewable ICRS image.

        Returns
        -------
        std : :py:class:`~imot_tools.io.s2image.Image`
            (N_level, N_height, N_width) standardized energy-levels.

        lsq : :py:class:`~imot_tools.io.s2image.Image`
            (N_level, N_height, N_width) least-squares energy-levels.
        """
        bfsf_grid = transform.pol2cart(
            1, self._synthesizer._grid_colat, self._synthesizer._grid_lon
        )
        icrs_grid = np.tensordot(self._synthesizer._R.T, bfsf_grid, axes=1)

        stat_std = self._statistics[0]
        field_std = self._synthesizer.synthesize(stat_std)
        std = image.Image(field_std, icrs_grid)

        stat_lsq = self._statistics[1]
        field_lsq = self._synthesizer.synthesize(stat_lsq)
        lsq = image.Image(field_lsq, icrs_grid)

        return std, lsq
