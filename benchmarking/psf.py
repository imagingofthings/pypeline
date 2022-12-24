import numpy as np, matplotlib.pyplot as plt
import astropy.units as u

from pypeline.phased_array import measurement_set

N_station = 62
N_pix = 2000

FoV = (N_pix*2*u.arcsec).to(u.rad).value
fname = '/project/c31/lofar30MHz1/LB_8hr/lofar30MHz1_t201806301100_SBL153.MS'

ms = measurement_set.LofarMeasurementSet(fname, N_station, station_only=True)
time = ms.time['TIME']

UVW_ms = ms.uvw

################## PSF from WSCLEAN
from astropy.io import fits

psf_clean = fits.getdata('/project/c31/lofar30MHz1/LB_8hr/lofar30MHz1-psf.fits', ext=0).squeeze()

plt.imshow(psf_clean)
plt.colorbar()
print(psf_clean.min(), psf_clean.max())
plt.savefig('psf_wsclean.png', bbox_inches='tight', facecolor='white')

################## Calculation of the PSF: numpy

from scipy import stats
from matplotlib.colors import LogNorm

UVW = UVW_ms.reshape(-1, 3)
FoV = (2000*2.*u.arcsec).to(u.rad).value

du = 2/FoV
dv = 2/FoV

bin_u = np.arange(-N_pix//2, N_pix//2+1, 1)*du
bin_v = np.arange(-N_pix//2, N_pix//2+1, 1)*dv

# count the number of visibility point in the uvw-plane
ret = stats.binned_statistic_2d(UVW[:, 0], UVW[:, 1], None, 'count', bins=[bin_u, bin_v]) 
Nuv = ret.statistic     # number of points per pixel
Suv = Nuv

psf_np = np.abs(np.fft.fftshift(np.fft.fft2(Suv)))
psf_np /= psf_np.max()
plt.imshow(psf_np)
plt.colorbar()
plt.savefig('psf_numpy.png', bbox_inches='tight', facecolor='white')

################## Calculation of the PSF: artificial point source
import bluebild_tools.cupy_util as bbt_cupy
use_cupy = bbt_cupy.is_cupy_usable()

import numpy as np, matplotlib.pyplot as plt

from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime

from scipy import constants
from scipy import stats

from imot_tools.io import fits as ifits, s2image

import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
from pypeline.phased_array.bluebild import data_processor as bb_dp, gram as bb_gr
from pypeline.phased_array.bluebild.imager import fourier_domain as bb_im
from pypeline.phased_array.data_gen import statistics
from pypeline.phased_array import measurement_set, beamforming, instrument
from pypeline.phased_array.data_gen import source

from pypeline.phased_array.data_gen import source

channel_id = 0
time_slice = 200 #7191
N_levels = 1
N_station = 62

# read MS
ms = measurement_set.LofarMeasurementSet(file_name=fname, N_station=N_station, station_only=True)
field_center = coord.SkyCoord(ra=1*u.rad, dec=1.57*u.rad, frame="icrs")
print(field_center)

src_config = [(coord.SkyCoord(ra=(field_center.ra.deg+1e-6)*u.deg, dec=(field_center.dec.deg+1e-6)*u.deg, frame="icrs"), 1)]
sky_model = source.SkyEmission(src_config)

# Observation quantities
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)

obs_start = atime.Time('2018-06-30T11:00:02.003', scale="utc", format="isot")
T_integration = 4.0055
time = atime.Time((obs_start + (T_integration * u.s) * np.arange(time_slice)).mjd, scale="utc", format="mjd")
print(len(time))

# Instrument & Imaging
dev = instrument.LofarBlock(N_station=N_station, station_only=True)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=np.inf)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)

gram = bb_gr.GramBlock()
eps = 1e-3
precision = 'single'
N_bits = 32

N_pix = 2000
N_eig, c_centroid = N_levels, list(range(N_levels))        # bypass centroids

I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq','sqrt'))
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV, field_center=field_center, eps=eps, n_trans=1, precision=precision)
px_grid = nufft_imager._synthesizer.xyz_grid

i_t = 0
for t in ProgressBar(time):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    S_corrected = IV_dp(D, V, W, c_idx)
    nufft_imager.collect(UVW_baselines_t, S_corrected)

lsq_image, sqrt_image = nufft_imager.get_statistic()

S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=N_pix, FoV=FoV, field_center=field_center, eps=eps, n_trans=1, precision=precision)

for t in ProgressBar(time):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    D, V = S_dp(XYZ, W, wl)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))
    nufft_imager.collect(UVW_baselines_t, S_sensitivity)

sensitivity_image = nufft_imager.get_statistic()[0]

I_lsq_eq  = s2image.Image(lsq_image.data  / sensitivity_image, px_grid)
I_sqrt_eq = s2image.Image(sqrt_image.data / sensitivity_image, px_grid)

psf_point = I_lsq_eq.data[0] / I_lsq_eq.data[0].max()
plt.imshow(psf_point)
plt.colorbar()
plt.savefig('psf_point.png', bbox_inches='tight', facecolor='white')


################## Calculation of the PSF: by setting the Visibilities to 1+0j
import finufft
from pypeline.util import frame
from imot_tools.io import s2image

precision = 'double'

_precision_mappings = dict(single=dict(complex=np.complex64, real=np.float32, dtype='float32'),
                           double=dict(complex=np.complex128, real=np.float64, dtype='float64'))

lim = np.sin(FoV / 2)
uvw_frame = frame.uvw_basis(field_center)
lmn_grid = np.tensordot(np.linalg.inv(uvw_frame), px_grid, axes=1).astype(_precision_mappings[precision]['real'])

lmn_grid = lmn_grid.reshape(3,-1)
_grid_center = lmn_grid.mean(axis=-1) 

# NUFFT synthetise variables
_n_trans, _eps = 1, 1e-3
typ_nufft = 3

UVW_baselines = UVW_ms.reshape(-1, 3)

UVW = (2 * np.pi * UVW_baselines / wl).astype(_precision_mappings[precision]['real'])
V = np.ones((UVW.shape[0])).astype(_precision_mappings[precision]['complex'])

if(typ_nufft == 3):
    plan = finufft.Plan(nufft_type=3, n_modes_or_dim=3, eps=_eps, isign=1, n_trans=_n_trans, dtype=_precision_mappings[precision]['dtype'])
    plan.setpts(x=UVW[:,0], y=UVW[:,1], z=UVW[:,2], s=lmn_grid[0], t=lmn_grid[1], u=lmn_grid[2])
elif(typ_nufft == 2):
    plan = finufft.Plan(nufft_type=2, n_modes_or_dim=3, eps=_eps, isign=1, n_trans=_n_trans, dtype=_precision_mappings[precision]['dtype'])
    plan.setpts(x=UVW[:,0], y=UVW[:,1], z=UVW[:,2], s=lmn_grid[0], t=lmn_grid[1], u=lmn_grid[2])
elif(typ_nufft == 1):
    plan = finufft.Plan(nufft_type=1, n_modes_or_dim=(N_pix, N_pix), eps=1e-4, isign=1)
    plan.setpts(x=2*lim/N_pix * UVW[:,0], y=2*lim/N_pix * UVW[:,1])  

Taper = np.exp(-(UVW[:,0]*UVW[:,0] + UVW[:,1]*UVW[:,1])/1e10)
Weight = np.sqrt(UVW[:,0]*UVW[:,0] + UVW[:,1]*UVW[:,1])

out = np.real(plan.execute(V))
#out = np.real(plan.execute(V * Taper))
#out = np.real(plan.execute(V * Weight))
#out = np.real(plan.execute(V * Weight * Taper))

img = out.reshape(N_pix, N_pix)

I_psf = s2image.Image(img, px_grid)

im = plt.imshow(I_psf.data[0])
plt.colorbar()
plt.savefig('psf_finufft.png', bbox_inches='tight', facecolor='white')

################## Compare the PSF from the point source and wsclean
def rad_average(data, cen_x=None, cen_y=None, bin_size=1):
    # Image center
    if(cen_x == None and cen_y == None):
        cen_y, cen_x = np.squeeze(np.where((data == data.max())))
    print(cen_y, cen_x)
    
    # Find radial distances
    [X, Y] = np.meshgrid(np.arange(data.shape[1]) - cen_x, np.arange(data.shape[0]) - cen_y)
    #R = np.sqrt(np.square(X) + np.square(Y))
    R = Y
    rad = np.arange(1, np.max(R), 1)
    intensity = np.zeros(len(rad))

    index= 0
    for i in rad:
        mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
        values = data[mask]
        intensity[index] = np.mean(values)
        index += 1
    return intensity, rad, cen_y, cen_x


def RescaleData(arr, a=0, b=1):
    scaled_arr = (arr.astype(np.float32) - np.min(arr))/(np.max(arr) - np.min(arr)) * (b-a) + a
    return scaled_arr


a, b = 10., 1000.
norm_psf_bb = RescaleData(psf_point, a, b)
norm_psf_clean = RescaleData(psf_clean, a, b)
#norm_psf_bb = psf_bb/np.mean(psf_bb)-1
#norm_psf_clean = psf_clean/np.mean(psf_clean)-1

psf_diff = norm_psf_bb / norm_psf_clean - 1

plt.figure(figsize=(10, 8), facecolor='white')
plt.imshow(psf_diff, origin='lower')
plt.colorbar()
#plt.xlabel('u [m]'), plt.ylabel('v [m]')
print(psf_diff.min(), psf_diff.max())

plt.figure()
intens, rad, y, x = rad_average(psf_point, bin_size=10)
norm_intens = RescaleData(intens)
plt.plot(rad, norm_intens, color='tab:orange')
#plt.show(), plt.clf()
print(rad.size)

intens, rad, _, _ = rad_average(psf_clean, bin_size=10)
norm_intens = RescaleData(intens)
plt.plot(rad, norm_intens, color='tab:blue')
#plt.show(), plt.clf()
print(rad.size)