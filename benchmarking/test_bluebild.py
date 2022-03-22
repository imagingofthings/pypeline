# #############################################################################
# test_blueblid.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com] (modified by Michele)
# Simulated LOFAR imaging with Bluebild (Standard and NUFT).
# #############################################################################
'''export OMP_NUM_THREADS=1''' 

from tqdm import tqdm as ProgressBar
import astropy.coordinates as coord
import astropy.time as atime
import astropy.units as u
from imot_tools.io import fits as ifits, s2image
import numpy as np
import cupy as cp
import scipy.constants as constants

import pypeline.phased_array.bluebild.gram as bb_gr

from pypeline.util import frame
from pypeline.phased_array import beamforming
from pypeline.phased_array.bluebild import gram as bb_gr, data_processor as bb_dp, parameter_estimator as bb_pe
from pypeline.phased_array.bluebild.imager import spatial_domain as bb_sd, fourier_domain as bb_im
from pypeline.phased_array.data_gen import source, statistics
from pypeline.phased_array import instrument
from timing import Timer

from pypeline.phased_array import measurement_set
from data_gen_utils import RandomDataGen, SimulatedDataGen, RealDataGen

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter('ignore', category=AstropyWarning)

t = Timer()

gpu = True
time_slice = 125
timeslice = slice(None,None,time_slice)

N_station = 24 # 60
N_pix = 512

N_level = 10
N_src = None

#path_out = '/users/mibianco/data/user_catalog/'
#path_out = '/users/mibianco/data/test/'
#fname = '/users/mibianco/data/psf/mock_catalog_psf.txt'
#fname = '/users/mibianco/data/psf/mock_catalog2.txt'
#fname_prefix = 'HTR_Nlvl%d_Nsrc%d' %(N_level, N_src)
#fname_prefix = 'psf'

path_out = '/users/mibianco/data/lofar_test/'
fname_prefix = 'lofar30MHz1'
path_in = '/users/mibianco/data/lofar/%s/' %fname_prefix
fname = '%slofar30MHz_t201806301100_SBL153.MS' %path_in

"""
path_out = '/users/mibianco/data/test_gauss4/'
path_in = '/users/mibianco/data/gauss4/'
fname_prefix = 'gauss4_losito'
fname = path_in + "gauss4_losito_t201806301100_SBL179.MS"

path_out = '/users/mibianco/data/gauss4/'
fname_prefix = 'gauss4'
path_in = '/users/mibianco/data/gauss4/'
fname = path_in + "gauss4_t201806301100_SBL180.MS"
"""

t.start_time("Set up data")
if('cat' in fname):
    # Observation
    obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
    field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
    FoV_deg = 5
    FoV, frequency = np.deg2rad(FoV_deg), 145e6
    wl = constants.speed_of_light / frequency
    T_integration = 8
    time = obs_start + (T_integration * u.s) * np.arange(3595)
    obs_end = time[-1]
    
    # Instrument
    dev = instrument.LofarBlock(N_station)
    mb_cfg = [(_, _, field_center) for _ in range(N_station)]
    mb = beamforming.MatchedBeamformerBlock(mb_cfg)
    gram = bb_gr.GramBlock()

    # Load catalog
    mock_catalog = np.loadtxt(fname)
    N_src = mock_catalog.ndim
    #sky_model = source.from_tgss_catalog(field_center, FoV, N_src=30)
    sky_model = source.user_defined_catalog(field_center, FoV, catalog_user=mock_catalog)
    vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=np.inf)  # SNR=np.inf (no noise)

elif('ms' in fname.lower()):
    # Measurement Set
    ms = measurement_set.LofarMeasurementSet(fname, N_station)
    channel_id = 1
    frequency = ms.channels["FREQUENCY"][channel_id]
    wl = constants.speed_of_light / frequency.to_value(u.Hz)

    # Observation
    #FoV = np.deg2rad(5)
    FoV = np.deg2rad(1.111111111)
    field_center = ms.field_center
    time = ms.time['TIME']

    #time_obs = ProgressBar(ms.visibilities(channel_id=[data.channel_id], time_id=slice(0, None, time_slice), column="DATA"))
    
    # Instrument
    dev = ms.instrument
    mb = ms.beamformer  
    gram = bb_gr.GramBlock()
else:
    ValueError('Parameter[type_data] is not valid. Please change to "ms" or "cat".')

# Imaging
eps = 1e-3
w_term = True
precision = 'single'
N_bits = 32

### Imaging parameters ===========================================================
if('cat' in fname):
    lim = np.sin(FoV / 2)
    grid_slice = np.linspace(-lim, lim, N_pix)
    l_grid, m_grid = np.meshgrid(grid_slice, grid_slice)
    n_grid = np.sqrt(1 - l_grid ** 2 - m_grid ** 2)  # No -1 if r on the sphere !
    lmn_grid = np.stack((l_grid, m_grid, n_grid), axis=0)
    uvw_frame = frame.uvw_basis(field_center)
    px_grid = np.tensordot(uvw_frame, lmn_grid, axes=1)
elif('ms' in fname.lower()):
    cl_WCS = ifits.wcs('%s%s-image.fits' %(path_in, fname_prefix))
    cl_WCS = cl_WCS.sub(['celestial']) 
    #cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10)))  # downsample, too high res!
    px_grid = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
    N_cl_lon, N_cl_lat = px_grid.shape[-2:]
    assert N_cl_lon == N_cl_lat
    N_pix = N_cl_lon

t.end_time("Set up data")
print('''You are running bluebild on file: %s
         with the following input parameters:
         %d timesteps
         %d stations
         clustering into %d levels
         The output grid will be %dx%d = %d pixels''' %(fname, len(time[timeslice]), N_station, N_level, px_grid.shape[1],  px_grid.shape[2],  px_grid.shape[1]* px_grid.shape[2]))

### Intensity Field =================================================
# Parameter Estimation
t.start_time("Estimate intensity field parameters")
"""
if(N_src == 1):
    N_eig, c_centroid = N_level, np.zeros(N_level)  #list(range(N_level))
else:
    I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
    for i_t, ti in enumerate(ProgressBar(time[::200])):
        if('ms' in fname.lower()):
            tobs, f, S = next(ms.visibilities(channel_id=[channel_id], time_id=slice(i_t, i_t+1, None), column="DATA"))
            wl = constants.speed_of_light / f.to_value(u.Hz) #self.wl
            XYZ = dev(tobs)
            W = mb(XYZ, wl)
            S, _ = measurement_set.filter_data(S, W)
        else:
            XYZ = dev(ti)
            W = mb(XYZ, wl)
            S = vis(XYZ, W, wl)
        
        G = gram(XYZ, W, wl)

        I_est.collect(S, G)
    N_eig, c_centroid = I_est.infer_parameters()
    print(N_eig, c_centroid)
"""
N_eig, c_centroid = N_level, list(range(N_level))        # bypass centroids
t.end_time("Estimate intensity field parameters")

####################################################################
#### Imaging
####################################################################
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq','sqrt'))
I_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)

UVW_baselines = []
gram_corrected_visibilities = []
for i_t, ti in enumerate(ProgressBar(time[:time_slice])):
    t.start_time("Synthesis: prep input matrices & fPCA")
    if('ms' in fname.lower()):
        tobs, f, S = next(ms.visibilities(channel_id=[channel_id], time_id=slice(i_t, i_t+1, None), column="DATA"))
        wl = constants.speed_of_light / f.to_value(u.Hz) #self.wl
        XYZ = dev(tobs)
        W = mb(XYZ, wl)
        S, _ = measurement_set.filter_data(S, W)
    else:
        XYZ = dev(ti)
        W = mb(XYZ, wl)
        S = vis(XYZ, W, wl)
    
    G = gram(XYZ, W, wl)
    D, V, c_idx = I_dp(S, G)
    c_idx = list(range(N_level))        # bypass c_idx
    t.end_time("Synthesis: prep input matrices & fPCA")
    
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

I_std_ss, I_lsq_ss = I_mfs_ss.as_image()

UVW_baselines = np.stack(UVW_baselines, axis=0).reshape(-1, 3)
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=-3).reshape(*S_corrected.shape[:2], -1)

# NUFFT Synthesis
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=px_grid, FoV=FoV, 
                                      field_center=field_center, eps=eps, w_term=w_term, 
                                      n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), 
                                      precision=precision)
#nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV, field_center=field_center, eps=eps, w_term=w_term, n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
lsq_image, sqrt_image = nufft_imager(gram_corrected_visibilities)
#============================================================================================

### Sensitivity Field =========================================================
t.start_time("Estimate sensitivity field parameters")
"""
# Parameter Estimation
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
"""
N_eig, c_centroid = N_level, list(range(N_level))
t.end_time("Estimate sensitivity field parameters")

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
S_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
sensitivity_coeffs = []

for i_t, ti in enumerate(ProgressBar(time[:time_slice])):
    if('ms' in fname.lower()):
        tobs, f, S = next(ms.visibilities(channel_id=[channel_id], time_id=slice(i_t, i_t+1, None), column="DATA"))
        wl = constants.speed_of_light / f.to_value(u.Hz) #self.wl
        XYZ = dev(tobs)
    else:
        XYZ = dev(ti)
    
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    D, V = S_dp(G)

    if(gpu):
        XYZ_gpu = cp.asarray(XYZ.data)
        W_gpu  = cp.asarray(W.data.toarray())
        V_gpu  = cp.asarray(V)
        _ = S_mfs_ss(D, V_gpu, XYZ_gpu, W_gpu, cluster_idx=np.zeros(N_eig, dtype=int))
    else:
        _ = S_mfs_ss(D, V, XYZ, W, cluster_idx=np.zeros(N_eig, dtype=int))

    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))  # (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    sensitivity_coeffs.append(S_sensitivity)

# Save eigen-values
np.save('%sD_%s' %(path_out, fname_prefix), D.reshape(-1, 1, 1))

_, S_ss = S_mfs_ss.as_image()

sensitivity_coeffs = np.stack(sensitivity_coeffs, axis=0).reshape(-1)
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=px_grid, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=1, precision=precision)
sensitivity_image = nufft_imager(sensitivity_coeffs)

# Image Gridding
I_lsq_eq_ss = s2image.Image(I_lsq_ss.data / S_ss.data, I_lsq_ss.grid)
#I_lsq_eq_ss = s2image.Image(I_lsq_ss.data / S_ss.data, px_grid)
#I_lsq_eq_ss = s2image.Image(I_lsq_ss.data / S_ss.data, nufft_imager._synthesizer.xyx_grid)
#I_sqrt_eq_nufft = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyx_grid)
I_lsq_eq_nufft = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)

# Save eigen-vectors for Standard Synthesis
np.save('%sI_ss_%s' %(path_out, fname_prefix), I_lsq_eq_ss.data)
#np.save('%sI_sqrt_eq_nufft_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_sqrt_eq_nufft.data)

# Interpolate image to MS grid-frame for Standard Synthesis
f_interp = (I_lsq_eq_ss.data.reshape(N_level, N_cl_lon, N_cl_lat).transpose(0, 2, 1))
I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
I_lsq_eq_interp.to_fits('%sI_ss_%s.fits' %(path_out, fname_prefix))

# Save eigen-vectors for NUFFT
#np.save('%sI_lsq_eq_nufft_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_lsq_eq_nufft.data)
np.save('%sI_nufft_%s' %(path_out, fname_prefix), I_lsq_eq_nufft.data)

# Interpolate image to MS grid-frame for NUFFT
f_interp = (I_lsq_eq_nufft.data.reshape(N_level, N_cl_lon, N_cl_lat).transpose(0, 2, 1))
I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
I_lsq_eq_interp.to_fits('%sI_nufft_%s.fits' %(path_out, fname_prefix))
