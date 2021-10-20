# #############################################################################
# lofar_bootes_ps_small_fov.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com] (modified by Matthieu)
# #############################################################################

"""
Simulated LOFAR imaging with Bluebild (PeriodicSynthesis).
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
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics
import pypeline.phased_array.instrument as instrument
import imot_tools.math.sphere.transform as transform
import time as tt
import pycsou.linop as pyclop
from imot_tools.math.func import SphericalDirichlet
import joblib as job
from timing import Timer

from other_utils import rad_average

t = Timer()

do_spherical_interpolation = True # BEWARE, if set to true the runtime becomes very slow!!
do_periodic_synthesis = True
timeslice = slice(None,None,5)

t.start_time("Set up data")


# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
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
#sky_model = source.from_tgss_catalog(field_center, FoV, N_src=30)
mock_catalog = np.array([[218.00001, 34.500001, 1e6]]) 
sky_model = source.user_defined_catalog(field_center, FoV, catalog_user=mock_catalog)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=30)
time = obs_start + (T_integration * u.s) * np.arange(3595)
obs_end = time[-1]

### Periodic Synthesis Imaging parameters ===========================================================
t1 = tt.time()
N_level = 4
N_bits = 32
R = dev.icrs2bfsf_rot(obs_start, obs_end)
_, _, pix_colat, pix_lon = grid.equal_angle(
    N=dev.nyquist_rate(wl),
    direction=R @ field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
    FoV=1.25 * FoV,
)
N_FS, T_kernel = dev.bfsf_kernel_bandwidth(wl, obs_start, obs_end), np.deg2rad(16)

### Standard Synthesis Imaging parameters ===========================================================
_, _, px_colat, px_lon = grid.equal_angle(
    N=dev.nyquist_rate(wl), direction=field_center.cartesian.xyz.value, FoV=1.25*FoV
)

px_grid = transform.pol2cart(1, px_colat, px_lon)

print('''You are running bluebild with the following input parameters:
         {0} timesteps
         {1} stations
         clustering into {2} levels
         The output grid will be {3}x{4} = {5} pixels'''.format(len(time[timeslice]), N_station,N_level, px_grid.shape[1],  px_grid.shape[2],  px_grid.shape[1]* px_grid.shape[2]))

### NUFFT imaging parameters ===========================================================
f_interp_ss = np.load("bluebild_ss_img")
f_interp_ps = np.save("bluebild_ps_img")
I_lsq_eq_ss_interp = s2image.Image(f_interp_ss, pix_xyz_interp)
I_lsq_eq_ps_interp = s2image.Image(f_interp_ps, pix_xyz_interp)

I_lsq_eq_ss_interp_data = I_lsq_eq_ss_interp.data
I_lsq_eq_ps_interp_data = I_lsq_eq_ps_interp.data
I_lsq_eq_nufft_data = np.save("bluebild_nufft_img").data

pix_xyz = np.save("bluebild_np_grid")

t2 = tt.time()
fig, ax = plt.subplots(ncols=3, nrows = 2, figsize=(16, 8))
ax = ax.flatten()
#I_lsq_eq_ss.draw(catalog=sky_model.xyz.T, ax=ax[0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[0].imshow(np.sum(I_lsq_eq_ss_interp, axis=0))
#I_lsq_eq_ss_interp.draw(catalog=sky_model.xyz.T, ax=ax[0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[0].set_title('Interpolated Standard Synthesis')

#I_lsq_eq_ps.draw(catalog=sky_model.xyz.T, ax=ax[1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[1].imshow(np.sum(I_lsq_eq_ps_interp, axis=0))
#I_lsq_eq_ps_interp.draw(catalog=sky_model.xyz.T, ax=ax[1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[1].set_title('Interpolated Periodic Synthesis')

#I_lsq_eq_nufft.draw(catalog=sky_model.xyz.T, ax=ax[2], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[2].imshow(np.sum(I_lsq_eq_nufft_data, axis=0))
ax[2].set_title('NUFFT')

intens, rad = rad_average(np.sum(I_lsq_eq_ss_interp, axis=0), bin_size=2)
ax[4].semilogy(rad, intens, color='b', label='SS')
intens, rad = rad_average(np.sum(I_lsq_eq_ps_interp, axis=0), bin_size=2)
ax[4].semilogy(rad, intens, color='g', label='PS')
intens, rad = rad_average(np.sum(I_lsq_eq_nufft_data, axis=0), bin_size=2)
ax[4].semilogy(rad, intens, color='r', label='NUFFT')
#np.save("bluebild_ps_img", f_interp_ps)
plt.savefig("%sdata/outputs/test_bluebild2" %(str(Path.home())+'/'), bbox_inches='tight')


"""
print(f'Elapsed time: {t2 - t1} seconds.')
print(f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180/np.pi)} degrees.\n'
      f'Run time {np.floor(t2 - t1)} seconds.')
fig, ax = plt.subplots(ncols=3, nrows = 2, figsize=(16, 8))
ax = ax.flatten()
#I_lsq_eq_ss.draw(catalog=sky_model.xyz.T, ax=ax[0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
I_lsq_eq_ss_interp = s2image.Image(f_interp_ss, pix_xyz_interp)
I_lsq_eq_ss_interp.draw(catalog=sky_model.xyz.T, ax=ax[0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[0].set_title('Interpolated Standard Synthesis')

#I_lsq_eq_ps.draw(catalog=sky_model.xyz.T, ax=ax[1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
I_lsq_eq_ps_interp = s2image.Image(f_interp_ps, pix_xyz_interp)
I_lsq_eq_ps_interp.draw(catalog=sky_model.xyz.T, ax=ax[1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[1].set_title('Interpolated Periodic Synthesis')

I_lsq_eq_nufft.draw(catalog=sky_model.xyz.T, ax=ax[2], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[2].set_title('NUFFT')

intens, rad = rad_average(np.sum(I_lsq_eq_ss_interp.data, axis=0), bin_size=2)
ax[4].semilogy(rad, intens, color='b', label='SS')
intens, rad = rad_average(np.sum(I_lsq_eq_ps_interp.data, axis=0), bin_size=2)
ax[4].semilogy(rad, intens, color='g', label='PS')
intens, rad = rad_average(np.sum(I_lsq_eq_nufft.data, axis=0), bin_size=2)
ax[4].semilogy(rad, intens, color='r', label='NUFFT')
#np.save("bluebild_ps_img", f_interp_ps)
plt.savefig("%sdata/outputs/test_bluebild2" %(str(Path.home())+'/'), bbox_inches='tight')
"""
np.save("bluebild_ss_img", f_interp_ss)
np.save("bluebild_ps_img", f_interp_ps)
np.save("bluebild_nufft_img", I_lsq_eq_nufft.data)
np.save("bluebild_np_grid", pix_xyz)

"""
t2 = tt.time()
print(f'Elapsed time: {t2 - t1} seconds.')
print(f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180/np.pi)} degrees.\n'
      f'Run time {np.floor(t2 - t1)} seconds.')
if do_spherical_interpolation:
    fig, ax = plt.subplots(ncols=3, nrows = 2, figsize=(16, 8))
else:
    fig, ax = plt.subplots(ncols=3, nrows = 1, figsize=(16, 8))
ax = ax.flatten()
I_lsq_eq_ss.draw(catalog=sky_model.xyz.T, ax=ax[0], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[0].set_title('Standard Synthesis')

I_lsq_eq_ps.draw(catalog=sky_model.xyz.T, ax=ax[1], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[1].set_title('Periodic Synthesis')

I_lsq_eq_nufft.draw(catalog=sky_model.xyz.T, ax=ax[2], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
ax[2].set_title('NUFFT')

if do_spherical_interpolation:
    I_lsq_eq_ss_interp = s2image.Image(f_interp_ss, pix_xyz_interp)
    I_lsq_eq_ss_interp.draw(catalog=sky_model.xyz.T, ax=ax[3], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
    ax[3].set_title('Interpolated Standard Synthesis')

    if do_periodic_synthesis:
        I_lsq_eq_ps_interp = s2image.Image(f_interp_ps, pix_xyz_interp)
        I_lsq_eq_ps_interp.draw(catalog=sky_model.xyz.T, ax=ax[4], data_kwargs=dict(cmap='cubehelix'), show_gridlines=False)
        ax[4].set_title('Interpolated Periodic Synthesis')
        
        np.save("bluebild_ps_img", f_interp_ps)
    plt.savefig("test_bluebild2")


    np.save("bluebild_ss_img", f_interp_ss)
    np.save("bluebild_nufft_img", I_lsq_eq_nufft.data)

else:
    plt.savefig("%sdata/outputs/test_bluebild_planes" %((str(Path.home())+'/')))
t.print_summary()
np.save("bluebild_np_grid", pix_xyz)
"""
#===
t.print_summary()