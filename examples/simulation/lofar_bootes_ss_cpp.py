# #############################################################################
# lofar_bootes_ss.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Simulated LOFAR imaging with Bluebild (StandardSynthesis).
"""
from tqdm import tqdm as ProgressBar
import astropy.coordinates as coord
import astropy.time as atime
import astropy.units as u
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import imot_tools.math.sphere.transform as transform
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import sys
import os
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics
import pypeline.phased_array.instrument as instrument
import json
from time import perf_counter

np.random.seed(0)

import bluebild
#print("bluebild from:", bluebild.__file__)
#print("sys.path", sys.path)

ctx = bluebild.Context(bluebild.ProcessingUnit.AUTO)


# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
FoV, frequency = np.deg2rad(5), 145e6
wl = constants.speed_of_light / frequency
first_time = True
N_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))

for N_bits in 32, 64:
#for N_bits in 32,:

    print("### N_bits =", N_bits)

    dtype_f = np.float32   if N_bits == 32 else np.float64
    dtype_c = np.complex64 if N_bits == 32 else np.complex128

    # Instrument
    N_station = 24  #24
    dev = instrument.LofarBlock(N_station)
    mb_cfg = [(_, _, field_center) for _ in range(N_station)]
    mb = beamforming.MatchedBeamformerBlock(mb_cfg)
    gram = bb_gr.GramBlock()

    # Data generation
    T_integration = 8
    sky_model = source.from_tgss_catalog(field_center, FoV, N_src=60)
    vis  = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=np.inf)
    vis2 = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=np.inf)
    time = obs_start + (T_integration * u.s) * np.arange(3595)
    time_slice = 2000 #2000

    # Imaging
    N_levels = 8 #24

    _, _, px_colat, px_lon = grid.equal_angle(N=dev.nyquist_rate(wl) * 2,
    #_, _, px_colat, px_lon = grid.equal_angle(N=int(dev.nyquist_rate(wl) / 16) + 1,
                                            direction=field_center.cartesian.xyz.value,
                                            FoV=FoV)
    px_grid = transform.pol2cart(1, px_colat, px_lon).astype(dtype=dtype_f)
    if first_time:
        print("px_grid:", type(px_grid), px_grid.ndim, len(px_grid), px_grid.size, px_grid.shape, px_grid.dtype)
        first_time = False

    ### Intensity Field ===========================================================
    # Parameter Estimation
    I_est = bb_pe.IntensityFieldParameterEstimator(N_levels, sigma=0.95)
    #for t in ProgressBar(time[::time_slice]):
    for t in time[::time_slice]:
        XYZ = dev(t)
        W = mb(XYZ, wl)
        S = vis(XYZ, W, wl)
        _ = vis2(XYZ, W, wl) #EO: keep this one to keep prng consistent later on
        G = gram(XYZ, W, wl)
        I_est.collect(S, G)    
    N_eig, c_centroid = I_est.infer_parameters()
    
    # Imaging
    #print("### PYTHON")

    XYZ_all    = []
    W_all_real = []
    W_all_imag = []
    D_all      = []
    V_all_real = []
    V_all_imag = []
    c_idx_all  = []

    I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid, ctx)
    I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_levels, N_bits)
    time_pyt = 0.0
    for t in time[::time_slice]:
        XYZ = dev(t)
        W = mb(XYZ, wl)
        S = vis(XYZ, W, wl)
        D, V, c_idx = I_dp(S, XYZ, W, wl)
        tic = perf_counter()
        stats_ref = I_mfs(D, V, XYZ.data, W.data, c_idx)
        time_pyt += (perf_counter() - tic)
        assert stats_ref.dtype == dtype_f, f"stats_ref {stats_ref.dtype} not of expected type {dtype_f}"

        XYZ_all.append(np.transpose(XYZ.data).astype(dtype_f, copy=False).tolist())
        W_all_real.append(np.transpose(W.data.real).astype(dtype_f, copy=False).tolist())
        W_all_imag.append(np.transpose(W.data.imag).astype(dtype_f, copy=False).tolist())
        V_all_real.append(np.transpose(V.real).astype(dtype_f, copy=False).tolist())
        V_all_imag.append(np.transpose(V.imag).astype(dtype_f, copy=False).tolist())
        D_all.append(D.data.tolist())
        c_idx_all.append(c_idx.tolist())
    
    #print(f"@@@Python {N_bits} {N_threads:2d} {time_pyt:.3f} sec")
        
    Nb, Ne = V.shape
    Na, Nb = W.shape
    Nc, Nh, Nw = px_grid.shape

    print(f"Na={Na} Nb={Nb} Nc={Nc} Ne={Ne} Nh={Nh} Nw={Nw}, sqrt(Npix)={int(np.sqrt(Nh*Nw))}")

    Data = {'wl': wl,
            'Na': Na, 'Nb': Nb, 'Nc': Nc, 'Ne': Ne, 'Nh': Nh, 'Nl': N_levels, 'Nw': Nw,
            'XYZ': XYZ_all,
            'W_real': W_all_real,
            'W_imag': W_all_imag,
            'D': D_all,
            'V_real': V_all_real,
            'V_imag': V_all_imag,
            'c_idx': c_idx_all,
            'grid': np.transpose(px_grid.data).tolist(),
            'statistics_std': np.transpose(I_mfs._statistics[0]).tolist(),
            'statistics_lsq': np.transpose(I_mfs._statistics[1]).tolist(),
            }

    with open(f"lofar_ss_{N_bits}.json", "w") as json_file:
        json.dump(Data, json_file, indent=4)
    print(f"Json dumped")

    ### C++ 
    I_mfs_cpp = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_levels, N_bits, ctx)
    time_cpp = 0.0
    for t in time[::time_slice]:
        XYZ = dev(t)
        W = mb(XYZ, wl)
        S = vis2(XYZ, W, wl)
        D, V, c_idx = I_dp(S, XYZ, W, wl)
        tic = perf_counter()
        stats = I_mfs_cpp(D, V, XYZ.data, W.data, c_idx)
        time_cpp += (perf_counter() - tic)
        if ctx is not None:
            assert I_mfs_cpp._stats_std.dtype == dtype_f, f"stats_std {stats_std.dtype} not of expected type {dtype_f}"

    #print(f"@@@ C++ {N_bits} {N_threads:2d} {time_cpp:.3f} sec")

    # Last epoch
    #print("last epoch max abs error on stats std =", np.max(np.absolute(stats[0] - stats_ref[0])))
    #print("last epoch max abs error on stats lsq =", np.max(np.absolute(stats[1] - stats_ref[1])))
    
    diff = I_mfs_cpp._stats_std_cum - I_mfs._statistics[0]
    mad_std  = np.max(np.abs(diff))
    rmse_std = np.sqrt(np.sum(diff**2)/np.size(diff))

    diff = I_mfs_cpp._stats_lsq_cum - I_mfs._statistics[1]
    mad_lsq  = np.max(np.abs(diff))
    rmse_lsq = np.sqrt(np.sum(diff**2)/np.size(diff))

    # Cumulated statistics
    print(f"@@@ {N_bits} {N_threads:2d} {time_pyt:.3f} {time_cpp:.3f} {time_pyt / time_cpp:.3f}; std error: RMSE ={rmse_std : .2E}, max abs ={mad_std : .2E}; lsq error: RMSE ={rmse_lsq : .2E}, max abs ={mad_lsq : .2E}")

    I_std,     I_lsq     = I_mfs.as_image()
    I_std_cpp, I_lsq_cpp = I_mfs_cpp.as_image()
    

    ### Sensitivity Field =========================================================
    # Parameter Estimation
    S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
    for t in ProgressBar(time[::time_slice]):
        XYZ = dev(t)
        W = mb(XYZ, wl)
        G = gram(XYZ, W, wl)

        S_est.collect(G)
    N_eig = S_est.infer_parameters()

    # Imaging
    S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
    S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits, ctx=None)
    for t in ProgressBar(time[::time_slice]):
        XYZ = dev(t)
        W = mb(XYZ, wl)
        G = gram(XYZ, W, wl)

        D, V = S_dp(XYZ, W, wl)
        _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
    _, S = S_mfs.as_image()


    # Plot Results ================================================================
    fig, ax = plt.subplots(nrows=2, ncols=2)

    I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid)
    I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax[0,0])
    ax[0,0].set_title("Bluebild Standardized Image")

    I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid)
    I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax[0,1])
    ax[0,1].set_title("Bluebild Least-Squares Image")

    I_std_eq_cpp = s2image.Image(I_std_cpp.data / S.data, I_std.grid)
    I_std_eq_cpp.draw(catalog=sky_model.xyz.T, ax=ax[1,0])
    ax[1,0].set_title("Bluebild Standardized Image C++")

    I_lsq_eq_cpp = s2image.Image(I_lsq_cpp.data / S.data, I_lsq.grid)
    I_lsq_eq_cpp.draw(catalog=sky_model.xyz.T, ax=ax[1,1])
    ax[1,1].set_title("Bluebild Least-Squares Image C++")


    fig.savefig("test.png")
    fig.show()
    plt.show()
