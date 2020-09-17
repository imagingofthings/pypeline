
import numpy as np
import sys
import scipy.constants as constants
import imot_tools.math.sphere.grid as grid
import imot_tools.math.sphere.transform as transform
import astropy.units as u
import astropy.time as atime
import astropy.coordinates as coord
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics
import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain as synth
import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain_optimized as synth_test
import timing
from dummy_synthesis import RandomDataGen, synthesize

'''
class RandomDataGen():
    def __init__(self):
        self.N_height  = 248
        self.N_width   = 124
        self.N_antenna = 550
        self.N_beam = 24
        self.N_eig  = 12
 
    def getPixGrid(self):
        return np.random.rand(3, self.N_height, self.N_width)*2-1

    def getV(self, i):
        V   = np.random.rand(self.N_beam, self.N_eig)-0.5 + 1j*np.random.rand(self.N_beam, self.N_eig)-0.5j
        return V

    def getXYZ(self, i):
        XYZ = np.random.rand(self.N_antenna,3)
        return XYZ

    def getW(self, i):
        W   = np.random.rand(self.N_antenna, self.N_beam)-0.5 + 1j*np.random.rand(self.N_antenna, self.N_beam)-0.5j
        return W

    def getVXYZW(self, i):
        return (self.getV(i),self.getXYZ(i),self.getW(i))'''


class SimulatedDataGen():
    def __init__(self, wl):
        self.wl = wl
        obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
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
        _, _, px_colat, px_lon = grid.equal_angle(
            N=self.dev.nyquist_rate(wl), direction=field_center.cartesian.xyz.value, FoV=FoV
        )
        self.pix_grid = transform.pol2cart(1, px_colat, px_lon)
        self.time = obs_start + (T_integration * u.s) * np.arange(3595)
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

    # timer
    timer = timing.Timer()


    ###### make simulated dataset ###### 
    # parameters
    frequency = 145e6
    wl = constants.speed_of_light / frequency
    precision = 32 # 32 or 64

    #data = SimulatedDataGen(wl)
    data = RandomDataGen()
    pix = data.getPixGrid()


    synthesizer_test = synth_test.SpatialFieldSynthesizerOptimizedBlock(wl, pix, precision)
    synthesizer      = synth.SpatialFieldSynthesizerBlock(wl, pix, precision)
    synthesizer_test.set_timer(timer, "Optimized ")
    synthesizer.set_timer(timer,)

    # strangely, whichever synthesizer is called first takes slightly more time

    for i in range(1,2):
        (V, XYZ, W) = data.getVXYZW(i)


        V1 = np.copy(V) # V gets modified by the synthesizer
        XYZ1 = np.copy(XYZ) # V gets modified by the synthesizer
        #timer.start_time("Run First Synthesizer")
        #stat_std = synthesizer(V1,XYZ,W)
        #timer.end_time("Run First Synthesizer")    
        timer.start_time("Run Test Synthesizer")
        stat_opt = synthesizer_test(V,XYZ,W)
        timer.end_time("Run Test Synthesizer")
        timer.start_time("Run Dummy Synthesizer")
        stat_dum = synthesize(pix,V1,XYZ1,W)
        timer.end_time("Run Dummy Synthesizer")

        print("Difference in results between dummy & optimized synthesizers:", np.average( stat_dum - stat_opt))
    print(timer.summary())