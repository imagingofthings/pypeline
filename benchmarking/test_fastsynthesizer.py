
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
        field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
        FoV = np.deg2rad(5)

        # Instrument
        N_station = 24  #24
        self.dev = instrument.LofarBlock(N_station)
        mb_cfg = [(_, _, field_center) for _ in range(N_station)]
        self.mb = beamforming.MatchedBeamformerBlock(mb_cfg)
        self.gram = bb_gr.GramBlock()

        # generation
        T_integration = 8
        sky_model = source.from_tgss_catalog(field_center, FoV, N_src=20)
        self.vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=np.inf)
        _, _, self.px_colat, self.px_lon = grid.equal_angle(
            N=self.dev.nyquist_rate(wl), direction=field_center.cartesian.xyz.value, FoV=FoV
        )
        self.pix_grid = transform.pol2cart(1, self.px_colat, self.px_lon)
        self.time = self.obs_start + (T_integration * u.s) * np.arange(3595)
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

    T_kernel = np.deg2rad(10)

    # random or simulated dataset
    data = SimulatedDataGen(wl)
    #data = dummy_synthesis.RandomDataGen()

    R = data.dev.icrs2bfsf_rot(data.obs_start, data.time[-1])
    N_FS = data.dev.bfsf_kernel_bandwidth(wl, data.obs_start, data.time[-1])

    ################################### 

    # timer
    timer = timing.Timer()
    # function to turn timing on or off
    #timer.off() 

    pix = data.getPixGrid()

    # The Fast Periodic Synthesis Kernel
    synthesizer_periodic  = synth_periodic.FourierFieldSynthesizerBlock(wl, data.px_colat, data.px_lon, N_FS, T_kernel, R, precision)
    synthesizer_periodic.set_timer(timer, "Periodic ")

    synthesizer_standard = synth_standard.SpatialFieldSynthesizerBlock(wl, pix, precision)
    synthesizer_standard.set_timer(timer, "Standard ")

    # iterate though timesteps
    # increase the range to run through more calls
    for t in range(1,2):
        (V, XYZ, W) = data.getVXYZW(t)
        print("t = {0}".format(t))


        #do some copying for inputs which get modified by the synthesizer
        V1 = np.copy(V) 
        XYZ1 = np.copy(XYZ) 
        V2 = np.copy(V)
        XYZ2 = np.copy(XYZ) 
        
        # call the Bluebild Synthesis Kernels
        stats_periodic = synthesizer_periodic(V,XYZ,W)
        stats_standard = synthesizer_standard(V,XYZ,W)

        # trasform the periodic field statistics to periodic eigenimages
        field_periodic = synthesizer_periodic.synthesize(stats_periodic)
        bfsf_grid = transform.pol2cart(
            1, synthesizer_periodic._grid_colat, synthesizer_periodic._grid_lon
        )
        icrs_grid = np.tensordot(synthesizer_periodic._R.T, bfsf_grid, axes=1)
        
        print("Difference in results between standard & periodic synthesizers:", np.average( stats_standard - field_periodic))

        img_standard = image.Image(stats_standard, pix)
        img_periodic = image.Image(field_periodic, icrs_grid)
        print("Difference between pix grid and  & icrs_grid:",  np.average(pix - icrs_grid))

        fig, ax = plt.subplots(ncols=2)
        img_standard.draw(ax=ax[0])
        ax[0].set_title("Bluebild Standard Image")

        img_periodic.draw(ax=ax[1])
        ax[1].set_title("Bluebild Periodic Image")
        fig.savefig("test_compare.png")
        fig.show()
        plt.show()

    print(timer.summary())