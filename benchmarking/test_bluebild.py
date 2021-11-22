# #############################################################################
# lofar_bootes_ps_small_fov.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com] (modified by Matthieu)
# #############################################################################

"""
Simulated LOFAR imaging with Bluebild (Standard, Periodic, and nufft).
"""

'''export OMP_NUM_THREADS=1''' 
import matplotlib as mpl
mpl.use('agg')
from pathlib import Path
# #############################################################################
# lofar_bootes_ps_small_fov.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com] (modified by Matthieu)
# #############################################################################

"""
Simulated LOFAR imaging with Bluebild (Standard, Periodic, and nufft).
"""

'''export OMP_NUM_THREADS=1''' 

from tqdm import tqdm as ProgressBar
import astropy.coordinates as coord
import astropy.time as atime
import astropy.units as u
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import scipy.constants as constants
import finufft

import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_fd
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_im
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics
import pypeline.phased_array.instrument as instrument
import imot_tools.math.sphere.transform as transform
import time as tt
import pycsou.linop as pyclop
from imot_tools.math.func import SphericalDirichlet
import joblib as job
from timing import Timer
from matplotlib import colors

from other_utils import nufft_make_grids

t = Timer()

gpu = True
do_spherical_interpolation = True # BEWARE, if set to true the runtime becomes very slow!!
do_periodic_synthesis = True
time_slice = 25
timeslice = slice(None,None,time_slice)

t.start_time("Set up data")

# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
FoV_deg = 5
FoV, frequency = np.deg2rad(FoV_deg), 145e6
wl = constants.speed_of_light / frequency

# Instrument
N_station = 24
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock()

# Data generation
T_integration = 8
#sky_model = source.from_tgss_catalog(field_center, FoV, N_src=30)
path_out = '/users/mibianco/data/user_catalog/'
#mock_catalog = np.array([[216.9, 32.8, 87.5], [218.2, 34.8, 87.5], [218.8, 32.8, 87.5], [217.8, 32.4, 87.5]]) 
mock_catalog = np.array([[218.00001, 34.500001, 1e6]])
N_src = mock_catalog.shape[0]
sky_model = source.user_defined_catalog(field_center, FoV, catalog_user=mock_catalog)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=30)
time = obs_start + (T_integration * u.s) * np.arange(3595)
obs_end = time[-1]

# Imaging
N_pix = 512
eps = 1e-3
w_term = True
precision = 'single'

### Periodic Synthesis Imaging parameters ===========================================================
t1 = tt.time()
N_level = 3
N_bits = 32
R = dev.icrs2bfsf_rot(obs_start, obs_end)
_, _, pix_colat, pix_lon = grid.equal_angle(
    N=dev.nyquist_rate(wl),
    direction=R @ field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
    FoV=1.25 * FoV)
N_FS, T_kernel = dev.bfsf_kernel_bandwidth(wl, obs_start, obs_end), np.deg2rad(16)

### Standard Synthesis Imaging parameters ===========================================================
_, _, px_colat, px_lon = grid.equal_angle(N=dev.nyquist_rate(wl), direction=field_center.cartesian.xyz.value, FoV=1.25*FoV)

px_grid = transform.pol2cart(1, px_colat, px_lon)

print('''You are running bluebild with the following input parameters:
         {0} timesteps
         {1} stations
         clustering into {2} levels
         The output grid will be {3}x{4} = {5} pixels'''.format(len(time[timeslice]), N_station,N_level, px_grid.shape[1],  px_grid.shape[2],  px_grid.shape[1]* px_grid.shape[2]))

### NUFFT imaging parameters ===========================================================

# Field center coordinates
field_center_lon, field_center_lat = field_center.data.lon.rad, field_center.data.lat.rad
field_center_xyz = field_center.cartesian.xyz.value

# UVW reference frame
w_dir = field_center_xyz
u_dir = np.array([-np.sin(field_center_lon), np.cos(field_center_lon), 0])
v_dir = np.array(
    [-np.cos(field_center_lon) * np.sin(field_center_lat), -np.sin(field_center_lon) * np.sin(field_center_lat),
     np.cos(field_center_lat)])
uvw_frame = np.stack((u_dir, v_dir, w_dir), axis=-1)

# Imaging grid
lim = np.sin(FoV / 2)
N_pix = 512
pix_slice = np.linspace(-lim, lim, N_pix)
Lpix, Mpix = np.meshgrid(pix_slice, pix_slice)
Jpix = np.sqrt(1 - Lpix ** 2 - Mpix ** 2)  # No -1 if r on the sphere !
lmn_grid = np.stack((Lpix, Mpix, Jpix), axis=0)
pix_xyz = np.tensordot(uvw_frame, lmn_grid, axes=1)
_, pix_lat_sq, pix_lon_sq = transform.cart2eq(*pix_xyz)

t.end_time("Set up data")

### Intensity Field =================================================
# Parameter Estimation
t.start_time("Estimate intensity field parameters")
if(N_src == 1):
    N_eig, c_centroid = N_level, np.zeros(N_level)  #list(range(N_level))
else:
    I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
    for ti in ProgressBar(time[::200]):
        XYZ = dev(ti)
        W = mb(XYZ, wl)
        S = vis(XYZ, W, wl)
        G = gram(XYZ, W, wl)

        I_est.collect(S, G)
    N_eig, c_centroid = I_est.infer_parameters()
t.end_time("Estimate intensity field parameters")

####################################################################
#### Imaging
####################################################################
_, px_grid_nufft = nufft_make_grids(FoV=FoV, grid_size=N_pix, field_center=field_center)    # get nufft grid sampling (copyed by pypeline/phased_array/bluebild/field_synthesizer/fourier_domain.py : self._make_grid())

I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq','sqrt'))
I_mfs_ps = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, N_level, N_bits)
#I_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)
I_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid_nufft, N_level, N_bits)

UVW_baselines = []
gram_corrected_visibilities = []
for ti in ProgressBar(time[::time_slice]):
    t.start_time("Synthesis: prep input matrices & fPCA")
    XYZ = dev(ti)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)

    D, V, c_idx = I_dp(S, G)
    t.end_time("Synthesis: prep input matrices & fPCA")
    t.start_time("Periodic Synthesis")
    _ = I_mfs_ps(D, V, XYZ.data, W.data, c_idx)
    t.end_time("Periodic Synthesis")

    t.start_time("Standard Synthesis")
    if(gpu):
        XYZ_gpu = cp.asarray(XYZ.data)
        W_gpu  = cp.asarray(W.data.toarray())
        V_gpu  = cp.asarray(V)
        _ = I_mfs_ss(D, V_gpu, XYZ_gpu, W_gpu, c_idx)
    else:
        _ = I_mfs_ss(D, V, XYZ.data, W.data, c_idx)
    t.end_time("Standard Synthesis")

    t.start_time("NUFFT Synthesis")
    UVW_baselines_t = dev.baselines(ti, uvw=True, field_center=field_center)
    UVW_baselines.append(UVW_baselines_t)
    S_corrected = IV_dp(D, V, W, c_idx)
    gram_corrected_visibilities.append(S_corrected)
    t.end_time("NUFFT Synthesis")

I_std_ps, I_lsq_ps = I_mfs_ps.as_image()
I_std_ss, I_lsq_ss = I_mfs_ss.as_image()

UVW_baselines = np.stack(UVW_baselines, axis=0).reshape(-1, 3)
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=-3).reshape(*S_corrected.shape[:2], -1)

# NUFFT Synthesis
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
lsq_image, sqrt_image = nufft_imager(gram_corrected_visibilities)
#lsq_image = nufft_imager(gram_corrected_visibilities)
#============================================================================================

### Sensitivity Field =========================================================
# Parameter Estimation
t.start_time("Estimate sensitivity field parameters")
if(N_src == 1):
    N_eig, c_centroid = N_level, np.zeros(N_level)  #list(range(N_level))
else:
    S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
    for ti in ProgressBar(time[::200]):
        XYZ = dev(ti)
        W = mb(XYZ, wl)
        G = gram(XYZ, W, wl)
        S_est.collect(G)
    N_eig, c_centroid = I_est.infer_parameters()
t.end_time("Estimate sensitivity field parameters")

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
S_mfs_ps = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, 1, N_bits)
#S_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits)
S_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid_nufft, 1, N_bits)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
sensitivity_coeffs = []
for ti in ProgressBar(time[::time_slice]): 
    XYZ = dev(ti)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    D, V = S_dp(G)

    _ = S_mfs_ps(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))

    if(gpu):
        XYZ_gpu = cp.asarray(XYZ.data)
        W_gpu  = cp.asarray(W.data.toarray())
        V_gpu  = cp.asarray(V)
        _ = S_mfs_ss(D, V_gpu, XYZ_gpu, W_gpu, cluster_idx=np.zeros(N_eig, dtype=int))
    else:
        _ = S_mfs_ss(D, V, XYZ, W, cluster_idx=np.zeros(N_eig, dtype=int))

    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))  # (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    sensitivity_coeffs.append(S_sensitivity)

np.save('%sD_ss_ps_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), D.reshape(-1, 1, 1))

_, S_ps = S_mfs_ps.as_image()
_, S_ss = S_mfs_ss.as_image()

I_lsq_eq_ps = s2image.Image(I_lsq_ps.data / S_ps.data, I_lsq_ps.grid)
#I_lsq_eq_ss = s2image.Image(I_lsq_ss.data / S_ss.data, I_lsq_ss.grid)
I_lsq_eq_ss = s2image.Image(I_lsq_ss.data / S_ss.data, px_grid_nufft)

sensitivity_coeffs = np.stack(sensitivity_coeffs, axis=0).reshape(-1)
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=1, precision=precision)
sensitivity_image = nufft_imager(sensitivity_coeffs)
#I_sqrt_eq_nufft = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
#np.save('%sI_sqrt_eq_nufft_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_sqrt_eq_nufft.data)
print(lsq_image.shape)
print(sensitivity_image.shape)
print(nufft_imager._synthesizer.xyz_grid.shape)
print((lsq_image/sensitivity_image).shape)
I_lsq_eq_nufft = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
print(I_lsq_eq_nufft.data.shape)
np.save('%sI_lsq_eq_nufft_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_lsq_eq_nufft.data)

### Spherical reinterpolation Field =========================================================
if do_spherical_interpolation:
    pix_xyz_interp = nufft_imager._synthesizer.xyz_grid
    #pix_xyz_interp = pix_xyz[:,::5, ::5]  # downsample, too high res!
    dirichlet_kernel = SphericalDirichlet(N=dev.nyquist_rate(wl), approx=True)

    nside = (dev.nyquist_rate(wl) + 1) / 3
    nodal_width = 2.8345 / np.sqrt(12 * nside ** 2)
    """
    interpolator_ss = pyclop.MappedDistanceMatrix(samples1=pix_xyz_interp.reshape(3, -1).transpose(), # output res
                                               samples2=px_grid.reshape(3, -1).transpose(), # input res
                                               function=dirichlet_kernel,
                                               mode='zonal', operator_type='sparse', max_distance=10 * nodal_width,
                                               #eps=1e-1,
                                               )

    with job.Parallel(backend='loky', n_jobs=-1, verbose=True) as parallel:
        interpolated_maps_ss = parallel(job.delayed(interpolator_ss)
                                     (I_lsq_eq_ss.data.reshape(N_level, -1)[n]) for n in range(N_level))

    f_interp_ss = np.stack(interpolated_maps_ss, axis=0).reshape((N_level,) + pix_xyz_interp.shape[1:])
    f_interp_ss = f_interp_ss / (dev.nyquist_rate(wl) + 1)
    f_interp_ss = np.clip(f_interp_ss, 0, None)
    
    I_lsq_eq_ss_interp = s2image.Image(f_interp_ss, pix_xyz_interp)
    np.save('%sI_lsq_eq_ss_interp_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_lsq_eq_ss_interp.data)
    """
    print('SHAPE:', I_lsq_eq_ss.data.shape)
    np.save('%sI_lsq_eq_ss_interp_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_lsq_eq_ss.data)

    #============================================================================================
    if do_periodic_synthesis:
        pix_bfsf = np.tensordot(R, pix_xyz_interp, axes=1)
        bb_pix_bfsf = transform.pol2cart(1, pix_colat, pix_lon)  # Bluebild critical support points

        interpolator_ps = pyclop.MappedDistanceMatrix(samples1=pix_bfsf.reshape(3, -1).transpose(), # output res
                                                   samples2=bb_pix_bfsf.reshape(3, -1).transpose(), # input res
                                                   function=dirichlet_kernel,
                                                   mode='zonal', operator_type='sparse', max_distance=10 * nodal_width,
                                                   #eps=1e-1,
                                                   )

        with job.Parallel(backend='loky', n_jobs=-1, verbose=True) as parallel:
            interpolated_maps_ps = parallel(job.delayed(interpolator_ps)
                                         (I_lsq_eq_ps.data.reshape(N_level, -1)[n])
                                         for n in range(N_level))

        f_interp_ps = np.stack(interpolated_maps_ps, axis=0).reshape((N_level,) + pix_bfsf.shape[1:])
        f_interp_ps = f_interp_ps / (dev.nyquist_rate(wl) + 1)
        f_interp_ps = np.clip(f_interp_ps, 0, None)

        I_lsq_eq_ps_interp = s2image.Image(f_interp_ps, pix_xyz_interp)
        np.save('%sI_lsq_eq_ps_interp_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_lsq_eq_ps_interp.data)
else:
    np.save('%sI_lsq_eq_ps_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_lsq_eq_ps.data)
    np.save('%sI_lsq_eq_ss_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_lsq_eq_ss.data)

#============================================================================================