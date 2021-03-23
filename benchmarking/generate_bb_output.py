import sys,timing
import numpy as np

import imot_tools.io.s2image as image
import imot_tools.math.sphere.transform as transform
import astropy.time as atime

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as synth_periodic
import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain as synth_standard

import dummy_synthesis 
from dummy_synthesis import synthesize, synthesize_stack

from data_gen_utils import RandomDataGen, SimulatedDataGen, RealDataGen

def draw_levels(stats_standard, field_periodic, stats_standard_norm, field_periodic_norm, pix, icrs_grid):
    grid_kwargs = {"ticks": False}
    img_standard = image.Image(stats_standard, pix)
    img_periodic = image.Image(field_periodic, icrs_grid)
    img_standard_norm = image.Image(stats_standard_norm, pix)
    img_periodic_norm = image.Image(field_periodic_norm, icrs_grid)

    fig, ax = plt.subplots(ncols=data.N_level+2, nrows = 2, figsize=(7, 3))
    #fig.tight_layout(pad = 2.0)
    img_standard.draw(ax=ax[0,0], data_kwargs = {"cmap": "Purples_r"}, grid_kwargs = grid_kwargs)
    ax[0,0].set_title("Standard Image\nAll Levels")
    img_periodic.draw(ax=ax[1,0], data_kwargs = {"cmap": "Purples_r"}, grid_kwargs = grid_kwargs)
    ax[1,0].set_title("Periodic Image\nAll Levels")
    img_standard_norm.draw(ax=ax[0,1], data_kwargs = {"cmap": "Greens_r"}, grid_kwargs = grid_kwargs)
    ax[0,1].set_title("Standard Image\nAll Levels, Normalized")
    img_periodic_norm.draw(ax=ax[1,1], data_kwargs = {"cmap": "Greens_r"}, grid_kwargs = grid_kwargs)
    ax[1,1].set_title("Periodic Image\nAll Levels, Normalized")
    for i in range(0,data.N_level):
        print(i)
        img_standard_norm.draw(ax=ax[0,i+2], index=i, data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
        ax[0,i+2].set_title("Standard Image\nNormalized Level {0}".format(i))
        img_periodic_norm.draw(ax=ax[1,i+2], index=i, data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
        ax[1,i+2].set_title("Periodic Image\nNormalized Level {0}".format(i))
    fig.show()
    plt.show()

def try_fix(stats_standard, stats_standard_norm,  pix, wcs,  thresfactor = 0.1):
    grid_kwargs = {"ticks": False}

    mask = np.sum(stats_standard, axis = 0)
    mask /= np.max(mask)

    for i in range(0,data.N_level):
       eigenimage = stats_standard_norm[i,:,:] 
       eigenimage *= mask
       thresh = np.max(eigenimage)*thresfactor
       #stats_standard_norm[i,:,:] = [0 if a_ > thresh else a_ for a_ in stats_standard_norm[i,:,:]]
       super_threshold_indices = eigenimage < thresh
       eigenimage[super_threshold_indices] = 0

    img_mask = image.Image(mask, pix)
    img_standard_norm = image.Image(stats_standard_norm, pix)
    #
    #img_mask = image.WCSImage(mask, wcs)
    i#mg_standard_norm = image.WCSImage(stats_standard_norm, wcs)

    import copy
    my_cmap = copy.copy(cm.get_cmap("GnBu_r")) # copy the default cmap
    my_cmap.set_bad((0,0,0))

    fig, ax = plt.subplots(ncols=data.N_level+2, nrows = 1, figsize=(8, 3))
    fig.tight_layout(pad = 0.5)
    img_mask.draw(ax=ax[0], data_kwargs = {"cmap": "Greys_r"}, grid_kwargs = grid_kwargs)
    ax[0].set_title("Mask Image\nAll Levels")
    img_standard_norm.draw(ax=ax[1], data_kwargs = {"cmap": my_cmap, "norm": LogNorm()}, grid_kwargs = grid_kwargs)
    ax[1].set_title("Standard Synthesis\nAlpha Mask\n{0}% Threshold\nAll Levels".format(thresfactor*100))
    for i in range(0,data.N_level):
        print(i)
        img_standard_norm.draw(ax=ax[i+2], index=i, data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
        ax[i+2].set_title("Standard Synthesis\nAlpha Mask\n{1}% Threshold\nLevel {0}".format(i,thresfactor*100))
    fig.show()
    plt.show()

    img_standard_norm.to_fits('bluebild_processed_4gauss.fits')

if __name__ == "__main__":

    ###### make simulated dataset ###### 

    precision = 32 # 32 or 64

    #data = SimulatedDataGen(frequency = 145e6)
    data = RealDataGen("/home/etolley/data/gauss4/gauss4_t201806301100_SBL180.MS", N_level = 4,  N_station = 24 ) # n level = # eigenimages
    #data = RealDataGen("/home/etolley/data/gauss2/gauss2_t201806301100_SBL180.MS", N_level = 2)
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
    stats_standard_combined = None
    stats_periodic_combined = None
    stats_standard_normcombined = None
    stats_periodic_normcombined = None
    icrs_grid = None
    for t in [0,1,50,51,100,101,200,201]:
        (V, Vs, XYZ, W, D, Ds) = data.getInputs(t)
        print("t = {0}".format(t))

        # call the Bluebild Synthesis Kernels
        stats_periodic = synthesizer_periodic(V,XYZ,W)
        stats_standard = synthesizer_standard(V,XYZ,W)
        stats_sens_periodic = synthesizer_periodic(Vs,XYZ,W)
        stats_sens_standard = synthesizer_standard(Vs,XYZ,W)
        stats_sens_standard = np.sum(stats_sens_standard, axis = 0)

        D_r  =  D.reshape(-1, 1, 1)
        #D_rs =  Ds.reshape(-1, 1, 1)

        stats_standard_norm = stats_standard * D_r
        stats_periodic_norm = stats_periodic * D_r
        #stats_sens_standard_norm = stats_sens_standard * D_rs
        #stats_sens_periodic_norm = stats_sens_periodic * D_rs

        # trasform the periodic field statistics to periodic eigenimages
        field_periodic      = synthesizer_periodic.synthesize(stats_periodic)
        field_periodic_norm = synthesizer_periodic.synthesize(stats_periodic_norm)
        field_sens_periodic      = synthesizer_periodic.synthesize(stats_sens_periodic)
        #field_sens_periodic_norm = synthesizer_periodic.synthesize(stats_sens_periodic_norm)


        bfsf_grid = transform.pol2cart(1, data.px_colat_periodic, data.px_lon_periodic)
        icrs_grid = np.tensordot(synthesizer_periodic._R.T, bfsf_grid, axes=1)

        try:
            print("Difference in results between standard & periodic synthesizers:", np.average( stats_standard - np.rot90(field_periodic,2)))
        except:
            print("Shapes are different between standard & periodic synthesizers. Standard: {0}, periodic: {1}".format(stats_standard.shape, field_periodic.shape ))
            print("Trimming down PS grid")
            icrs_grid                = icrs_grid[:,:,:-1]
            field_periodic           = field_periodic[:,:,:-1]
            field_periodic_norm      = field_periodic_norm[:,:,:-1]
            field_sens_periodic      = field_sens_periodic[:,:,:-1]
            #field_sens_periodic_norm = field_sens_periodic_norm[:,:,:-1]

        print(stats_standard.shape)
        print(stats_sens_standard.shape)
        try:
            for n in range(stats_standard.shape[0]):
                stats_standard_combined[n,:,:]     += stats_standard[n,:,:]/stats_sens_standard
                stats_standard_normcombined[n,:,:] += stats_standard_norm[n,:,:]/stats_sens_standard
                stats_periodic_combined[n,:,:]     += field_periodic[n,:,:]/stats_sens_standard
                stats_periodic_normcombined[n,:,:] += field_periodic_norm[n,:,:]/stats_sens_standard
        except:
            stats_standard_combined = np.zeros(stats_standard.shape)
            stats_standard_normcombined = np.zeros(stats_standard.shape)
            stats_periodic_combined = np.zeros(stats_standard.shape)
            stats_periodic_normcombined = np.zeros(stats_standard.shape)
            for n in range(stats_standard.shape[0]):
                stats_standard_combined[n,:,:]     += stats_standard[n,:,:]/stats_sens_standard
                stats_standard_normcombined[n,:,:] += stats_standard_norm[n,:,:]/stats_sens_standard
                stats_periodic_combined[n,:,:]     += field_periodic[n,:,:]/stats_sens_standard
                stats_periodic_normcombined[n,:,:] += field_periodic_norm[n,:,:]/stats_sens_standard

        #draw_comparison(stats_standard, field_periodic, pix, icrs_grid)
    #draw_levels(stats_standard_combined, stats_periodic_combined,
    #            stats_standard_normcombined, stats_periodic_normcombined, pix, icrs_grid)
    from astropy.io import fits
    import astropy.wcs as pywcs
    with fits.open("/home/etolley/data/gauss4/gauss4-image-pb.fits") as hdul:
    #with fits.open("/home/etolley/data/gauss4/C_4gaussian-model.fits") as hdul:
        wcs = pywcs.WCS(hdul[0].header)
    img_standard = image.Image(stats_standard_combined, pix)
    img_periodic = image.Image(stats_periodic_combined, icrs_grid)
    img_standard.to_fits('bluebild_standard_4gauss.fits')
    img_periodic.to_fits('bluebild_periodic_4gauss.fits')

    try_fix(stats_standard_combined, stats_standard_normcombined,  pix, wcs)

    print(timer.summary())