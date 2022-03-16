
import matplotlib as mpl
mpl.use('agg')
from pathlib import Path

from tqdm import tqdm as ProgressBar
import astropy.coordinates as coord
import astropy.time as atime
import astropy.units as u
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import imot_tools.io.fits as ifits
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
import pypeline.phased_array.measurement_set as measurement_set
from data_gen_utils import RandomDataGen, SimulatedDataGen, RealDataGen

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter('ignore', category=AstropyWarning)

t = Timer()

gpu = True
time_slice = 125
timeslice = slice(None,None,time_slice)

N_pix = 512

N_level = 4
N_src = 4

#path_out = '/users/mibianco/data/user_catalog/'
#path_out = '/users/mibianco/data/psf/Npix_%d_noiseless/' %N_pix
#path_out = '/users/mibianco/data/test/'
#filename = '/users/mibianco/data/psf/mock_catalog_psf.txt'
#filename = '/users/mibianco/data/psf/mock_catalog2.txt'
#cname = 'HTR_Nlvl%d_Nsrc%d' %(N_level, N_src)
#cname = 'psf'

path_out = '/users/mibianco/data/lofar_test/'
cname = 'lofar30MHz256'
#path_in = '/users/mibianco/data/lofar/lofar30MHz_256/'
path_in = '/users/mibianco/data/lofar/lofar30MHz_256/'
filename = path_in + "lofar30MHz_t201806301100_SBH256.MS"

"""
path_out = '/users/mibianco/data/test_gauss4/'
path_in = '/users/mibianco/data/gauss4/'
cname = 'gauss4_losito'
filename = path_in + "gauss4_losito_t201806301100_SBL179.MS"

path_out = '/users/mibianco/data/gauss4/'
cname = 'gauss4'
path_in = '/users/mibianco/data/gauss4/'
filename = path_in + "gauss4_t201806301100_SBL180.MS"
"""

t.start_time("Set up data")
if('cat' in filename):
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
    N_station = 24
    dev = instrument.LofarBlock(N_station)
    mb_cfg = [(_, _, field_center) for _ in range(N_station)]
    mb = beamforming.MatchedBeamformerBlock(mb_cfg)
    gram = bb_gr.GramBlock()

    # Load catalog
    mock_catalog = np.loadtxt(filename)
    N_src = mock_catalog.ndim
    #sky_model = source.from_tgss_catalog(field_center, FoV, N_src=30)
    sky_model = source.user_defined_catalog(field_center, FoV, catalog_user=mock_catalog)
    vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=np.inf)  # SNR=np.inf (no noise)

elif('ms' in filename.lower()):
    # Measurement Set
    N_src = None
    N_station = 24 # 60

    ms = measurement_set.LofarMeasurementSet(filename, N_station)
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

t1 = tt.time()

### NUFFT imaging parameters ===========================================================
if('cat' in filename):
    _, px_grid_nufft = nufft_make_grids(FoV=FoV, grid_size=N_pix, field_center=field_center)    # get nufft grid sampling (copyed by pypeline/phased_array/bluebild/field_synthesizer/fourier_domain.py : self._make_grid())
elif('ms' in filename.lower()):
    cl_WCS = ifits.wcs('%s%s-psf.fits' %(path_in, cname))
    cl_WCS = cl_WCS.sub(['celestial']) 
    #cl_WCS = cl_WCS.slice((slice(None, None, 10), slice(None, None, 10)))  # downsample, too high res!
    px_grid_nufft = ifits.pix_grid(cl_WCS)  # (3, N_cl_lon, N_cl_lat) ICRS reference frame
    N_cl_lon, N_cl_lat = px_grid_nufft.shape[-2:]
    assert N_cl_lon == N_cl_lat
    N_pix = N_cl_lon

t.end_time("Set up data")
print('''You are running bluebild on file: %s
         with the following input parameters:
         %d timesteps
         %d stations
         clustering into %d levels
         The output grid will be %dx%d = %d pixels''' %(filename, len(time[timeslice]), N_station, N_level, px_grid_nufft.shape[1],  px_grid_nufft.shape[2],  px_grid_nufft.shape[1]* px_grid_nufft.shape[2]))

### Intensity Field =================================================
# Parameter Estimation

t.start_time("Estimate intensity field parameters")
"""
if(N_src == 1):
    N_eig, c_centroid = N_level, np.zeros(N_level)  #list(range(N_level))
else:
    I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
    for i_t, ti in enumerate(ProgressBar(time[::200])):
        if('ms' in filename.lower()):
            tobs, f, S = next(data.ms.visibilities(channel_id=[data.channel_id], time_id=slice(i_t, i_t+1, None), column="DATA"))
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
    #N_eig, c_centroid = I_est.infer_parameters()
    print(N_eig, c_centroid)
"""
N_eig, c_centroid = N_level, list(range(N_level))        # bypass centroids
t.end_time("Estimate intensity field parameters")

####################################################################
#### Imaging
####################################################################

I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq','sqrt'))

#I_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)
I_mfs_ss = bb_sd.Spatial_IMFS_Block(wl, px_grid_nufft, N_level, N_bits)

UVW_baselines = []
gram_corrected_visibilities = []
#for i_t, ti in enumerate(ProgressBar(time[::time_slice])):
for i_t, ti in enumerate(ProgressBar(time[:time_slice])):

    t.start_time("Synthesis: prep input matrices & fPCA")
    if('ms' in filename.lower()):
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
    
    t.start_time("NUFFT Synthesis")
    UVW_baselines_t = dev.baselines(ti, uvw=True, field_center=field_center)
    UVW_baselines.append(UVW_baselines_t)
    S_corrected = IV_dp(D, V, W, c_idx)
    gram_corrected_visibilities.append(S_corrected)
    t.end_time("NUFFT Synthesis")


UVW_baselines = np.stack(UVW_baselines, axis=0).reshape(-1, 3)
gram_corrected_visibilities = np.stack(gram_corrected_visibilities, axis=-3).reshape(*S_corrected.shape[:2], -1)

# NUFFT Synthesis
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=px_grid_nufft, FoV=FoV, field_center=field_center, eps=eps, w_term=w_term, n_trans=np.prod(gram_corrected_visibilities.shape[:-1]), precision=precision)
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
    #N_eig, c_centroid = I_est.infer_parameters()
"""
N_eig, c_centroid = N_level, list(range(N_level))
t.end_time("Estimate sensitivity field parameters")

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
sensitivity_coeffs = []

for i_t, ti in enumerate(ProgressBar(time[:time_slice])):
    if('ms' in filename.lower()):
        tobs, f, S = next(ms.visibilities(channel_id=[channel_id], time_id=slice(i_t, i_t+1, None), column="DATA"))
        wl = constants.speed_of_light / f.to_value(u.Hz)
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

    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))  # (W @ ((V @ np.diag(D)) @ V.transpose().conj())) @ W.transpose().conj()
    sensitivity_coeffs.append(S_sensitivity)

#np.save('%sD_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), D.reshape(-1, 1, 1))
np.save('%sD_%s' %(path_out, cname), D.reshape(-1, 1, 1))

sensitivity_coeffs = np.stack(sensitivity_coeffs, axis=0).reshape(-1)
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, UVW=UVW_baselines.T, grid_size=N_pix, FoV=FoV,
                                      field_center=field_center, eps=eps, w_term=w_term,
                                      n_trans=1, precision=precision)
sensitivity_image = nufft_imager(sensitivity_coeffs)

#I_sqrt_eq_nufft = s2image.Image(sqrt_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
#np.save('%sI_sqrt_eq_nufft_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_sqrt_eq_nufft.data)
I_lsq_eq_nufft = s2image.Image(lsq_image / sensitivity_image, nufft_imager._synthesizer.xyz_grid)


f_interp = (I_lsq_eq_nufft.data.reshape(N_level, N_cl_lon, N_cl_lat).transpose(0, 2, 1))
I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
I_lsq_eq_interp.to_fits('%sI_nufft_%s.fits' %(path_out, cname))

#np.save('%sI_lsq_eq_nufft_Nsrc%d_Nlvl%d' %(path_out, N_src, N_level), I_lsq_eq_nufft.data)
np.save('%sI_nufft_%s' %(path_out, cname), I_lsq_eq_nufft.data)
