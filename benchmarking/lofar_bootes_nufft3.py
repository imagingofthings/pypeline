# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
Simulation LOFAR imaging with Bluebild (NUFFT).
"""
import matplotlib as mpl
mpl.use('agg')
from pathlib import Path

from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime
import imot_tools.io.s2image as s2image
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import finufft
from imot_tools.io.plot import cmap
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as bb_synth
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_im
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.instrument as instrument
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.data_gen.statistics as statistics
from mpl_toolkits.mplot3d import Axes3D
import imot_tools.io.s2image as im
import imot_tools.io.plot as implt
import time as tt

from matplotlib import colors

# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")
FoV, frequency = np.deg2rad(1), 145e6
wl = constants.speed_of_light / frequency

# Instrument
N_station = 24
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock()

# Data generation
T_integration = 8
#sky_model = source.from_tgss_catalog(field_center, FoV, N_src=40)
path_out = '/users/mibianco/data/user_catalog/'
mock_catalog = np.array([[218.00001, 34.500001, 1e6]])
#mock_catalog = np.array([[216.9, 32.8, 87.5], [218.2, 34.8, 87.5], [218.8, 32.8, 87.5], [217.8, 32.4, 87.5]]) 
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

t1 = tt.time()
N_level = 4
time_slice = 25

### Intensity Field ===========================================================
# Parameter Estimation
"""
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S = vis(XYZ, W, wl)
    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()
"""

N_eig, c_centroid = N_level, np.zeros(N_level) #list(range(N_level))

# Imaging
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))# 'sqrt'))
UVW_baselines = []
gram_corrected_visibilities = []

for t in ProgressBar(time[::time_slice]):
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    UVW_baselines.append(UVW_baselines_t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)
    D, V, c_idx = I_dp(S, G)
    c_idx = list(range(N_level))        # bypass centroids
    S_corrected = IV_dp(D, V, W, c_idx)
    gram_corrected_visibilities.append(S_corrected)

print(S_corrected.shape)
print(np.shape(gram_corrected_visibilities))
print(np.ndim(gram_corrected_visibilities))
UVW_baselines = np.stack(UVW_baselines, axis=0).reshape(-1, 3)
print(UVW_baselines.shape)
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=-3).reshape(*S_corrected.shape[:2], -1)
print(gram_corrected_visibilities.shape)

# NUFFT Synthesis
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
print(nufft_imager._synthesizer._inner_fft_sizes)
#lsq_image, sqrt_image = nufft_imager(gram_corrected_visibilities)
lsq_image = nufft_imager(gram_corrected_visibilities)

### Sensitivity Field =========================================================
# Parameter Estimation
"""
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()
"""
N_eig, c_centroid = N_level, list(range(N_level))

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
sensitivity_coeffs = []
for t in ProgressBar(time[::time_slice]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    D, V = S_dp(G)
    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))
    sensitivity_coeffs.append(S_sensitivity)

sensitivity_coeffs = np.stack(sensitivity_coeffs, axis=0).reshape(-1)
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=1, precision=precision)
print(nufft_imager._synthesizer._inner_fft_sizes)
sensitivity_image = nufft_imager(sensitivity_coeffs)

print(lsq_image.shape)
print(sensitivity_image.shape)
print((lsq_image / sensitivity_image).shape)
print(nufft_imager._synthesizer.xyz_grid.shape)

I_lsq_eq = s2image.Image(lsq_image.squeeze() / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
#I_sqrt_eq = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
#np.save('%sbluebild_nufft_img' %path_out, I_lsq_eq.data)
np.save('%sD_nufft_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), D.reshape(-1, 1, 1))
np.save('%sI_lsq_eq_nufft_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_lsq_eq.data)
print('SHAPE:', I_lsq_eq.data.shape)
t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')

"""
fig, ax = plt.subplots(ncols=2, nrows = 1, figsize=(10, 8))
my_ext = [-N_pix//2, N_pix//2, -N_pix//2, N_pix//2]

ax = ax.flatten()
#I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax[0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
im = ax[0].imshow(I_lsq_eq.data[0], cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
#ax[0].set_title(f'Bluebild least-squares, sensitivity-corrected image (NUFFT)\n' f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n' f'Run time {np.floor(t2 - t1)} seconds.')
ax[0].set_title('BB lsq NUFFT')
fig.colorbar(im, ax=ax[0], orientation='vertical', pad=0.01, fraction=0.048)

I_sqrt_eq.draw(catalog=sky_model.xyz.T, ax=ax[1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
im = ax[1].imshow(I_sqrt_eq.data[0], cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
#ax[1].set_title(f'Bluebild sqrt, sensitivity-corrected image (NUFFT)\n' f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n' f'Run time {np.floor(t2 - t1)} seconds.')
ax[1].set_title('BB sqrt NUFFT')
fig.colorbar(im, ax=ax[1], orientation='vertical', pad=0.01, fraction=0.048)

plt.savefig("%stest_nufft1" %path_out, bbox_inches='tight')

plt.figure(figsize=(10,8))
titles = ['Strong sources', 'Mild sources', 'Faint Sources']
for i in range(lsq_image.shape[0]):
    plt.subplot(1, N_level, i + 1)
    ax = plt.gca()
    plt.title(titles[i])
    I_lsq_eq.draw(index=i, catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'),
                  catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5), show_gridlines=False)

plt.suptitle(f'Bluebild Eigenmaps')
plt.savefig("%stest_nufft3" %path_out)
#t.print_summary()
"""