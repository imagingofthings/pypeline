
import numpy as np
import sys
import scipy.constants as constants
import imot_tools.math.sphere.grid as grid
import imot_tools.io.s2image as image
import imot_tools.math.sphere.transform as transform
import astropy.units as u
import astropy.time as atime
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics
import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as synth_periodic
import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain as synth_standard
import timing
import dummy_synthesis 
from dummy_synthesis import RandomDataGen, synthesize, synthesize_stack

class SimulatedDataGen():
    def __init__(self, wl):
        self.wl = wl
        self.obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
        T_integration = 8
        self.time = self.obs_start + (T_integration * u.s) * np.arange(3595)
        field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
        self.FoV = np.deg2rad(5)

        # Instrument
        N_station = 24  #24
        self.dev = instrument.LofarBlock(N_station)
        mb_cfg = [(_, _, field_center) for _ in range(N_station)]
        self.mb = beamforming.MatchedBeamformerBlock(mb_cfg)
        self.gram = bb_gr.GramBlock()

        # generation

        self.R = self.dev.icrs2bfsf_rot(self.obs_start, self.time[-1])
        self.sky_model = source.from_tgss_catalog(field_center, self.FoV, N_src=5)
        self.vis = statistics.VisibilityGeneratorBlock(self.sky_model, T_integration, fs=196000, SNR=np.inf)

        _, _, self.px_colat_periodic, self.px_lon_periodic = grid.equal_angle(
            N=self.dev.nyquist_rate(wl),
            direction= self.R @ field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
            FoV=self.FoV
        )

        _, _, self.px_colat, self.px_lon = grid.equal_angle(
            N=self.dev.nyquist_rate(wl),
            direction=field_center.cartesian.xyz.value,
            FoV=self.FoV
        )
        self.pix_grid = transform.pol2cart(1, self.px_colat, self.px_lon)

        self.i = 0

        self.estimateParams()

    def estimateParams(self):
        # Parameter Estimation
        I_est = bb_pe.IntensityFieldParameterEstimator(4, sigma=0.95)
        for t in self.time[::200]:
            XYZ = self.dev(t)
            W = self.mb(XYZ, self.wl)
            S = self.vis(XYZ, W, self.wl)
            G = self.gram(XYZ, W, self.wl)

            I_est.collect(S, G)

        N_eig, c_centroid = I_est.infer_parameters()
        self.I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)

    def getPixGrid(self):
        return self.pix_grid
    def getVXYZW(self, i):
        try:
            t = self.time[i]
        except:
            print(i,"is an invalid time index")

        XYZ = self.dev(t)
        W = self.mb(XYZ, self.wl)
        S = self.vis(XYZ, W, self.wl)
        G = self.gram(XYZ, W, self.wl)
        __, V, __ = self.I_dp(S, G)
        return (V,XYZ.data, W.data)


if __name__ == "__main__":

    ###### make simulated dataset ###### 
    # parameters
    frequency = 145e6
    obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
    obs_end = atime.Time(56880.54171302732, scale="utc", format="mjd")
    wl = constants.speed_of_light / frequency
    precision = 32 # 32 or 64


    # random or simulated dataset
    data = SimulatedDataGen(wl)
    #data = dummy_synthesis.RandomDataGen()
    T_kernel = np.deg2rad(10)


    N_FS = data.dev.bfsf_kernel_bandwidth(wl, data.obs_start, data.time[-1])
    print("Default N_FS:", N_FS)
    N_FS = 50001

    ################################### 

    # timer
    timer = timing.Timer()
    # function to turn timing on or off
    #timer.off() 

    pix = data.getPixGrid()

    # The Fast Periodic Synthesis Kernel
    synthesizer_periodic  = synth_periodic.FourierFieldSynthesizerBlock(wl, data.px_colat_periodic, data.px_lon_periodic, N_FS, T_kernel, data.R, precision)
    synthesizer_periodic.set_timer(timer, "Periodic ")

    synthesizer_standard = synth_standard.SpatialFieldSynthesizerBlock(wl, pix, precision)
    synthesizer_standard.set_timer(timer, "Standard ")

    # iterate though timesteps
    # increase the range to run through more calls
    for t in range(1,2):
        (V, XYZ, W) = data.getVXYZW(t)
        print("t = {0}".format(t))

        # call the Bluebild Synthesis Kernels
        stats_periodic = synthesizer_periodic(V,XYZ,W)
        stats_standard = synthesizer_standard(V,XYZ,W)

        # trasform the periodic field statistics to periodic eigenimages
        field_periodic = synthesizer_periodic.synthesize(stats_periodic)

        icrs_grid = np.tensordot(synthesizer_periodic._R.T, pix, axes=1)
        
        print("Difference in results between standard & periodic synthesizers:", np.average( stats_standard - np.rot90(field_periodic,2)))

        img_standard  = image.Image(stats_standard, pix)
        img_periodic_rot  = image.Image(np.rot90(field_periodic,2), pix)
        img_diff_pix = image.Image(stats_standard - np.rot90(field_periodic,2), pix)

        img_standard_rot = image.Image(np.rot90(stats_standard,2), icrs_grid)
        img_periodic = image.Image(field_periodic, icrs_grid)
        img_diff_icrs_grid = image.Image(np.rot90(stats_standard,2) - field_periodic, icrs_grid)

        print("Difference between pix grid and  & icrs_grid:",  np.average(pix - icrs_grid))

        fig, ax = plt.subplots(ncols=3, nrows = 2)
        plotcolors ="GnBu_r"
        diffcolors = "RdBu"
        fig.tight_layout(pad = 2.0)
        img_standard.draw(catalog=data.sky_model.xyz.T, ax=ax[0,0], data_kwargs = {"cmap": plotcolors})
        ax[0,0].set_title("Bluebild Standard Image\nSS Grid")
        img_periodic_rot.draw(ax=ax[0,1], data_kwargs = {"cmap": plotcolors})
        ax[0,1].set_title("Bluebild Periodic Image\nRotated 180$^\circ$\nSS Grid")
        img_diff_pix.draw(ax=ax[0,2], data_kwargs = {"cmap": diffcolors})
        ax[0,2].set_title("Difference\nSS Grid")

        img_standard_rot.draw(ax=ax[1,0], data_kwargs = {"cmap": plotcolors})
        ax[1,0].set_title("Bluebild Standard Image\nRotated 180$^\circ$\nPS Grid")
        img_periodic.draw(ax=ax[1,1], data_kwargs = {"cmap": plotcolors})
        ax[1,1].set_title("Bluebild Periodic Image\nPS Grid")
        img_diff_icrs_grid.draw(ax=ax[1,2], data_kwargs = {"cmap": diffcolors})
        ax[1,2].set_title("Difference\nPS Grid")

        fig.savefig("test_compare.png")
        fig.show()
        plt.show()

    print(timer.summary())