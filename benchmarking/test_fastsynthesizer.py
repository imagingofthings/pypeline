import sys,timing
import numpy as np

import imot_tools.io.s2image as image
import imot_tools.math.sphere.transform as transform
import astropy.time as atime
import matplotlib.pyplot as plt

import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as synth_periodic
import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain as synth_standard

import dummy_synthesis 
from dummy_synthesis import synthesize, synthesize_stack

from data_gen_utils import RandomDataGen, SimulatedDataGen, RealDataGen

if __name__ == "__main__":

    ###### make simulated dataset ###### 

    precision = 32 # 32 or 64

    #data = SimulatedDataGen(frequency = 145e6)
    data = RealDataGen("/home/etolley/data/gauss4/gauss4_t201806301100_SBL180.MS")
    data.N_level = 4
    #data = dummy_synthesis.RandomDataGen()

    ################################### 

    # timer
    timer = timing.Timer()
    # function to turn timing on or off
    #timer.off() 

    pix = data.getPixGrid()

    # The Fast Periodic Synthesis Kernel
    synthesizer_periodic  = synth_periodic.FourierFieldSynthesizerBlock(data.wl, data.px_colat_periodic, data.px_lon_periodic, data.N_FS, data.T_kernel, data.R, precision)
    synthesizer_periodic.set_timer(timer, "Periodic ")

    synthesizer_standard = synth_standard.SpatialFieldSynthesizerBlock(data.wl, pix, precision)
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

        bfsf_grid = transform.pol2cart(1, data.px_colat_periodic, data.px_lon_periodic)
        icrs_grid = np.tensordot(synthesizer_periodic._R.T, bfsf_grid, axes=1)


        if pix.shape != icrs_grid.shape:
            print("Trimming down PS grid")
            icrs_grid = icrs_grid[:,:,:-1]
            field_periodic = field_periodic[:,:,:-1]
        
        try:
            print("Difference in results between standard & periodic synthesizers:", np.average( stats_standard - np.rot90(field_periodic,2)))
        except:
            print("Shapes are different between standard & periodic synthesizers. Standard: {0}, periodic: {1}".format(stats_standard.shape, field_periodic.shape ))

        print(stats_standard.shape, pix.shape)
        print(field_periodic.shape, icrs_grid.shape)
        print(field_periodic[:,10,10], stats_standard[:,10,10])
        img_standard  = image.Image(stats_standard, pix)
        img_periodic = image.Image(field_periodic, icrs_grid)
        img_periodic_rot  = image.Image(np.rot90(field_periodic,2), pix)
        img_standard_rot = image.Image(np.rot90(stats_standard,2), icrs_grid)


        img_diff_pix = image.Image(stats_standard - np.rot90(field_periodic,2), pix)
        img_diff_icrs_grid = image.Image(np.rot90(stats_standard,2) - field_periodic, icrs_grid)

        print("Difference between pix grid and  & icrs_grid:",  np.average(pix - icrs_grid))

        fig, ax = plt.subplots(ncols=3, nrows = 2)
        fig.tight_layout(pad = 2.0)
        grid_kwargs = {"ticks": False}
        color = "GnBu_r"
        color2 = "PuBu_r"
        color_diff = "RdBu"
        img_standard.draw(ax=ax[0,0], data_kwargs = {"cmap": color}, grid_kwargs = grid_kwargs)
        ax[0,0].set_title("Bluebild Standard Image\nSS Grid")
        img_periodic_rot.draw(ax=ax[0,1], data_kwargs = {"cmap": color}, grid_kwargs = grid_kwargs)
        ax[0,1].set_title("Bluebild Periodic Image\nRotated 180$^\circ$\nSS Grid")
        img_diff_pix.draw(ax=ax[0,2], data_kwargs =  {"cmap": color_diff},grid_kwargs = grid_kwargs)
        ax[0,2].set_title("Difference\nSS Grid")

        img_standard_rot.draw(ax=ax[1,0], data_kwargs = {"cmap": color}, grid_kwargs = grid_kwargs)
        ax[1,0].set_title("Bluebild Standard Image\nRotated 180$^\circ$\nPS Grid")
        img_periodic.draw(ax=ax[1,1], data_kwargs = {"cmap": color}, grid_kwargs = grid_kwargs)
        ax[1,1].set_title("Bluebild Periodic Image\nPS Grid")
        img_diff_icrs_grid.draw(ax=ax[1,2], data_kwargs =  {"cmap": color_diff}, grid_kwargs = grid_kwargs)
        ax[1,2].set_title("Difference\nPS Grid")


        fig.savefig("test_compare.png")
        fig.show()
        plt.show()

        fig, ax = plt.subplots(ncols=data.N_level, nrows = 2)
        for i in range(data.N_level):
            img_standard.draw(ax=ax[0,i], index=slice(0+i,1+i,None), data_kwargs = {"cmap": color}, grid_kwargs = grid_kwargs)
            ax[0,i].set_title("Bluebild Standard Image Level {0}".format(i))
            img_periodic.draw(ax=ax[1,i], index=slice(0+i,1+i,None), data_kwargs = {"cmap": color2}, grid_kwargs = grid_kwargs)
            ax[1,i].set_title("Bluebild Periodic Image Level {0}".format(i))
        fig.show()
        plt.show()

    print(timer.summary())