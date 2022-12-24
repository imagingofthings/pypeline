<<<<<<< HEAD
import matplotlib as mpl
mpl.use('agg')
from pathlib import Path
=======

import os
if os.getenv('OMP_NUM_THREADS') == None : os.environ['OMP_NUM_THREADS'] = "1"

import bluebild_tools.cupy_util as bbt_cupy
use_cupy = bbt_cupy.is_cupy_usable()
>>>>>>> ci-master

import sys,timing
import numpy as np
import scipy.sparse as sparse
import imot_tools.io.s2image as image
import imot_tools.math.sphere.transform as transform
import astropy.time as atime
import matplotlib.pyplot as plt
<<<<<<< HEAD
import imot_tools.io.s2image as s2image

=======
>>>>>>> ci-master
import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as synth_periodic
import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain as synth_standard
import dummy_synthesis 
from dummy_synthesis import synthesize, synthesize_stack
from data_gen_utils import RandomDataGen, SimulatedDataGen, RealDataGen
from other_utils import rad_average
path = '/users/mibianco/data/user_catalog/'


# For CuPy agnostic code
# ----------------------
xp = bbt_cupy.cupy if use_cupy else np


def draw_comparison(stats_standard, field_periodic, pix, icrs_grid):
    img_standard = image.Image(stats_standard, pix)
    img_periodic = image.Image(field_periodic, icrs_grid)
    img_periodic_rot = image.Image(np.rot90(field_periodic,2), pix)
    img_standard_rot = image.Image(np.rot90(stats_standard,2), icrs_grid)
    img_diff_pix = image.Image(stats_standard - np.rot90(field_periodic,2), pix)
    img_diff_icrs_grid = image.Image(np.rot90(stats_standard,2) - field_periodic, icrs_grid)

    print("Difference between pix grid and  & icrs_grid:",  np.average(pix - icrs_grid))

    fig, ax = plt.subplots(ncols=3, nrows = 2)
    fig.tight_layout(pad = 2.0)
    grid_kwargs = {"ticks": False}
    color_diff = "RdBu"
    img_standard.draw(ax=ax[0,0], data_kwargs = {"cmap": "GnBu_r"}, grid_kwargs = grid_kwargs)
    ax[0,0].set_title("Bluebild Standard Image\nSS Grid")
    img_periodic_rot.draw(ax=ax[0,1], data_kwargs = {"cmap": "GnBu_r"}, grid_kwargs = grid_kwargs)
    ax[0,1].set_title("Bluebild Periodic Image\nRotated 180$^\circ$\nSS Grid")
    img_diff_pix.draw(ax=ax[0,2], data_kwargs =  {"cmap": color_diff},grid_kwargs = grid_kwargs)
    ax[0,2].set_title("Difference\nSS Grid")

    img_standard_rot.draw(ax=ax[1,0], data_kwargs = {"cmap": "GnBu_r"}, grid_kwargs = grid_kwargs)
    ax[1,0].set_title("Bluebild Standard Image\nRotated 180$^\circ$\nPS Grid")
    img_periodic.draw(ax=ax[1,1], data_kwargs = {"cmap": "GnBu_r"}, grid_kwargs = grid_kwargs)
    ax[1,1].set_title("Bluebild Periodic Image\nPS Grid")
    img_diff_icrs_grid.draw(ax=ax[1,2], data_kwargs =  {"cmap": color_diff}, grid_kwargs = grid_kwargs)
    ax[1,2].set_title("Difference\nPS Grid")
    fig.savefig("%sdata/outputs/test_compare_Nsrc%d_Nlvl%d.png" %(str(Path.home())+'/', data.N_sources, data.N_level), bbox_inches='tight')
    #fig.show()
    #plt.show()

def draw_levels(stats_standard, field_periodic, stats_standard_norm, field_periodic_norm, pix, icrs_grid, psf):
    grid_kwargs = {"ticks": False}
    img_standard = image.Image(stats_standard, pix)
    img_periodic = image.Image(field_periodic, icrs_grid)
    img_standard_norm = image.Image(stats_standard_norm, pix)
    np.save('%sbluebild_ss_Nsrc%d_Nlvl%d' %(path, data.N_sources, data.N_level), img_standard_norm.data)
    img_periodic_norm = image.Image(field_periodic_norm, icrs_grid)

    if(psf):
        fig, ax = plt.subplots(ncols=data.N_level+2, nrows = 3, figsize=(15, 8))
    else:
        fig, ax = plt.subplots(ncols=data.N_level+2, nrows = 2, figsize=(15, 5))
    #fig.tight_layout(pad = 2.0)
    img_standard.draw(ax=ax[0,0], data_kwargs = {"cmap": "Purples_r"}, grid_kwargs = grid_kwargs)
    ax[0,0].set_title("SS")
    img_periodic.draw(ax=ax[1,0], data_kwargs = {"cmap": "Purples_r"}, grid_kwargs = grid_kwargs)
    ax[1,0].set_title("PS")
    img_standard_norm.draw(ax=ax[0,1], data_kwargs = {"cmap": "Greens_r"}, grid_kwargs = grid_kwargs)
    ax[0,1].set_title("SS Normalized")
    img_periodic_norm.draw(ax=ax[1,1], data_kwargs = {"cmap": "Greens_r"}, grid_kwargs = grid_kwargs)
    ax[1,1].set_title("PS Normalized")

    if(psf):
        # PSF radial distribution profile
        intens, rad = rad_average(np.sum(img_standard.data, axis=0), bin_size=2)
        ax[2,0].semilogy(rad, intens, color='b', label='standard')
        intens, rad = rad_average(np.sum(img_periodic.data, axis=0), bin_size=2)
        ax[2,0].semilogy(rad, intens, color='g', label='periodic')
        ax[2,0].legend()
        intens, rad = rad_average(np.sum(img_standard_norm.data, axis=0), bin_size=2)
        ax[2,1].semilogy(rad, intens, color='b')
        intens, rad = rad_average(np.sum(img_periodic_norm.data, axis=0), bin_size=2)
        ax[2,1].semilogy(rad, intens, color='g')
    for i in range(0,data.N_level):
        print(i)
        print(img_standard_norm.data.shape)
        img_standard_norm.draw(ax=ax[0,i+2], index=i, data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
        ax[0,i+2].set_title("SS Normalized\nLevel {0}".format(i))
        img_periodic_norm.draw(ax=ax[1,i+2], index=i, data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
        ax[1,i+2].set_title("PS Normalized\nLevel {0}".format(i))
        if(psf):
            intens, rad = rad_average(img_standard_norm.data[i], bin_size=2)
            ax[2,i+2].semilogy(rad, intens, color='b')
            intens, rad = rad_average(img_periodic_norm.data[i], bin_size=2)
            ax[2,i+2].semilogy(rad, intens, color='g')
    plt.subplots_adjust(wspace=0.5)
    fig.savefig("%stest_Nsrc%d_Nlvl%d.png" %(path, data.N_sources, data.N_level), bbox_inches='tight')
    #fig.show()
    #plt.show()



def draw_levels_substr(stats_standard, field_periodic, stats_standard_norm, field_periodic_norm, pix, icrs_grid, psf):
    grid_kwargs = {"ticks": False}
    img_standard = image.Image(stats_standard, pix)
    img_periodic = image.Image(field_periodic, icrs_grid)
    img_standard_norm = image.Image(stats_standard_norm, pix)
    img_periodic_norm = image.Image(field_periodic_norm, icrs_grid)

<<<<<<< HEAD
    img_standard_substr = img_standard_norm.data[0]
    for i in range(1,img_standard_norm.data.shape[0]):
        img_standard_substr -= img_standard_norm.data[i]
    img_periodic_substr = img_periodic_norm.data[0]
    for i in range(1,img_periodic_norm.data.shape[0]):
        img_periodic_substr -= img_periodic_norm.data[i]

    img_standard_substr = s2image.Image(img_standard_substr, pix)
    img_periodic_substr = s2image.Image(img_periodic_substr, pix)
    #np.save('%sbluebild_ss_Nsrc%d_Nlvl%d' %(path, data.N_sources, data.N_level), img_standard_norm.data)

    if(psf):
        fig, ax = plt.subplots(ncols=3, nrows = 3, figsize=(10, 8))
    else:
        fig, ax = plt.subplots(ncols=3, nrows = 2, figsize=(15, 5))
    #fig.tight_layout(pad = 2.0)
    img_standard.draw(ax=ax[0,0], data_kwargs = {"cmap": "Purples_r"}, grid_kwargs = grid_kwargs)
    ax[0,0].set_title("SS")
    img_periodic.draw(ax=ax[1,0], data_kwargs = {"cmap": "Purples_r"}, grid_kwargs = grid_kwargs)
    ax[1,0].set_title("PS")
    img_standard_norm.draw(ax=ax[0,1], data_kwargs = {"cmap": "Greens_r"}, grid_kwargs = grid_kwargs)
    ax[0,1].set_title("SS Normalized")
    img_periodic_norm.draw(ax=ax[1,1], data_kwargs = {"cmap": "Greens_r"}, grid_kwargs = grid_kwargs)
    ax[1,1].set_title("PS Normalized")

    if(psf):
        # PSF radial distribution profile
        intens, rad = rad_average(np.sum(img_standard.data, axis=0), bin_size=2)
        ax[2,0].semilogy(rad, intens, color='b', label='SS')
        intens, rad = rad_average(np.sum(img_periodic.data, axis=0), bin_size=2)
        ax[2,0].semilogy(rad, intens, color='g', label='PS')
        ax[2,0].legend()
        intens, rad = rad_average(np.sum(img_standard_norm.data, axis=0), bin_size=2)
        ax[2,1].semilogy(rad, intens, color='b')
        intens, rad = rad_average(np.sum(img_periodic_norm.data, axis=0), bin_size=2)
        ax[2,1].semilogy(rad, intens, color='g')
    
    #im = ax[0,2].imshow(img_standard_substr, cmap='Blues', origin='lower')
    #fig.colorbar(im, ax=ax[0,2], orientation='vertical', pad=0.01, fraction=0.048)
    img_standard_substr.draw(ax=ax[0,2], data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
    ax[0,2].set_title("SS Norm Substr")
    intens, rad = rad_average(img_standard_substr.data, bin_size=2)
    ax[2,2].semilogy(rad, intens, color='b')
    ax[1,2].set_title("PS Norm Substr")
    
    #im = ax[1,2].imshow(img_periodic_substr, cmap='Blues', origin='lower')
    #fig.colorbar(im, ax=ax[1,2], orientation='vertical', pad=0.01, fraction=0.048)
    img_periodic_substr.draw(ax=ax[1,2], data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
    intens, rad = rad_average(img_periodic_substr.data, bin_size=2)
    ax[2,2].semilogy(rad, intens, color='g')

    plt.subplots_adjust(wspace=0.5)
    fig.savefig("%stest_Nsrc%d_Nlvl%d_substr.png" %(path, data.N_sources, data.N_level), bbox_inches='tight')
    #fig.show()
    #plt.show()


=======
>>>>>>> ci-master
def draw_standard_levels(stats_standard, stats_standard_norm, pix):
    grid_kwargs = {"ticks": False}
    img_standard = image.Image(stats_standard, pix)
    img_standard_norm = image.Image(stats_standard_norm, pix)

    fig, ax = plt.subplots(ncols=data.N_level+2, nrows = 1, figsize=(7, 3))
    #fig.tight_layout(pad = 2.0)
    img_standard.draw(ax=ax[0,0], data_kwargs = {"cmap": "Purples_r"}, grid_kwargs = grid_kwargs)
    ax[0,0].set_title("Standard Image\nAll Levels")
    #img_periodic.draw(ax=ax[1,0], data_kwargs = {"cmap": "Purples_r"}, grid_kwargs = grid_kwargs)
    ax[1,0].set_title("Periodic Image\nAll Levels")
    img_standard_norm.draw(ax=ax[0,1], data_kwargs = {"cmap": "Greens_r"}, grid_kwargs = grid_kwargs)
    ax[0,1].set_title("Standard Image\nAll Levels, Normalized")
    #img_periodic_norm.draw(ax=ax[1,1], data_kwargs = {"cmap": "Greens_r"}, grid_kwargs = grid_kwargs)
    ax[1,1].set_title("Periodic Image\nAll Levels, Normalized")
    for i in range(0,data.N_level):
        print(i)
        img_standard_norm.draw(ax=ax[0,i+2], index=i, data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
        ax[0,i+2].set_title("Standard Image\nNormalized Level {0}".format(i))
        #img_periodic_norm.draw(ax=ax[1,i+2], index=i, data_kwargs = {"cmap": "Blues_r"}, grid_kwargs = grid_kwargs)
        ax[1,i+2].set_title("Periodic Image\nNormalized Level {0}".format(i))
    fig.savefig("%stest_standard_Nsrc%d_Nlvl%d.png" %(path, data.N_sources, data.N_level), bbox_inches='tight')

    #fig.show()
    #plt.show()



if __name__ == "__main__":

    ###### make simulated dataset ###### 

    precision = 32 # 32 or 64

    #data = SimulatedDataGen(frequency = 145e6)
<<<<<<< HEAD
    #data = RealDataGen("/users/mibianco/data/gauss4/gauss4_t201806301100_SBL180.MS", N_level=4, N_station=24) # n level = # eigenimages
    #cat = np.array([[216.9, 32.8, 87.5], [218.2, 34.8, 87.5], [218.8, 32.8, 87.5], [217.8, 32.4, 87.5]]) 
    #cat = np.array([[216.9, 32.8, 190.2]]) 
    cat = np.array([[218.00001, 34.500001, 1e6]]) 
    data = SimulatedDataGen(frequency=145e6, N_level=4 , N_sources=1, mock_catalog=cat)
=======
    data = RealDataGen("/work/scitas-share/SKA/data/gauss4/gauss4_t201806301100_SBL180.MS", N_level = 4, N_station = 24) # n level = # eigenimages
>>>>>>> ci-master
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

    for t in range(0,10):
        (V, XYZ, W, D) = data.getVXYZWD(t)
        print("t = {0}".format(t))
        
        if isinstance(W, sparse.csr.csr_matrix) or isinstance(W, sparse.csc.csc_matrix):
            W = W.toarray()

        print("use_cupy =", use_cupy)

        if use_cupy:
            XYZ = xp.asarray(XYZ)
            W   = xp.asarray(W)
            V   = xp.asarray(V)

        stats_standard = synthesizer_standard(V, XYZ, W)
        print("stats_standard = ", type(stats_standard))
        stats_periodic = synthesizer_periodic(V, XYZ, W)
        print("stats_periodic = ", type(stats_periodic))

        # call the Bluebild Synthesis Kernels
        #stats_standard = synthesizer_standard(V,XYZ,W)

        D_r =  D.reshape(-1, 1, 1)

        stats_standard_norm = stats_standard * D_r
        stats_periodic_norm = stats_periodic * D_r

        # transform the periodic field statistics to periodic eigenimages
        field_periodic      = synthesizer_periodic.synthesize(stats_periodic)
        field_periodic_norm = synthesizer_periodic.synthesize(stats_periodic_norm)

        bfsf_grid = transform.pol2cart(1, data.px_colat_periodic, data.px_lon_periodic)
        icrs_grid = np.tensordot(synthesizer_periodic._R.T, bfsf_grid, axes=1)

        try:
            print("Difference in results between standard & periodic synthesizers:", np.average( stats_standard - np.rot90(field_periodic,2)))
        except:
            print("Shapes are different between standard & periodic synthesizers. Standard: {0}, periodic: {1}".format(stats_standard.shape, field_periodic.shape ))
            print("Trimming down PS grid")
            icrs_grid = icrs_grid[:,:,:-1]
            field_periodic = field_periodic[:,:,:-1]
            field_periodic_norm = field_periodic_norm[:,:,:-1]

        try:    stats_standard_combined += stats_standard
        except: stats_standard_combined = stats_standard
        try:    stats_periodic_combined += field_periodic
        except: stats_periodic_combined = field_periodic
        try:    stats_standard_normcombined += stats_standard_norm
        except: stats_standard_normcombined = stats_standard_norm
        try:    stats_periodic_normcombined += field_periodic_norm
        except: stats_periodic_normcombined = field_periodic_norm

    #draw_comparison(stats_standard, field_periodic, pix, icrs_grid)
    draw_levels(stats_standard_combined, stats_periodic_combined, stats_standard_normcombined, stats_periodic_normcombined, pix, icrs_grid, psf=True)
    draw_levels_substr(stats_standard_combined, stats_periodic_combined, stats_standard_normcombined, stats_periodic_normcombined, pix, icrs_grid, psf=True)

    #draw_standard_levels(stats_standard_combined, stats_standard_normcombined, pix)
   
    print(timer.summary())
