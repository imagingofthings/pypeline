import matplotlib as mpl
mpl.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from other_utils import rad_average, RescaleData
from astropy.io import fits

def draw_levels(I_ss, I_ps, I_nufft, psf=True):
    if(psf):
        fig, ax = plt.subplots(ncols=I_ss.shape[0]+1, nrows=4, figsize=(15, 8))
    else:
        fig, ax = plt.subplots(ncols=I_ss.shape[0]+1, nrows=3, figsize=(15, 5))
    fig.tight_layout(pad = 2.0)
    
    my_ext = [-I_ss.shape[-1]//2, I_ss.shape[-1]//2, -I_ss.shape[-1]//2, I_ss.shape[-1]//2]

    ax[0,0].set_title("SS")
    im = ax[0,0].imshow(np.sum(I_ss, axis=0), cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[0,0], orientation='vertical', pad=0.01, fraction=0.048)

    ax[1,0].set_title("PS")
    im = ax[1,0].imshow(np.sum(I_ps, axis=0), cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[1,0], orientation='vertical', pad=0.01, fraction=0.048)

    ax[2,0].set_title("NUFFT")
    im = ax[2,0].imshow(np.sum(I_nufft, axis=0), cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[2,0], orientation='vertical', pad=0.01, fraction=0.048)
    print(I_nufft.shape)
    if(psf):
        # PSF radial distribution profile
        intens, rad = rad_average(np.sum(I_ss, axis=0), bin_size=2)
        ax[3,0].semilogy(rad, intens, color='b', label='SS')
        intens, rad = rad_average(np.sum(I_ps, axis=0), bin_size=2)
        ax[3,0].semilogy(rad, intens, color='g', label='PS')
        intens, rad = rad_average(np.sum(I_nufft, axis=0), bin_size=2)
        ax[3,0].semilogy(rad, intens, color='r', label='NUFFT')
        ax[3,0].legend()
        """
        intens, rad = rad_average(np.sum(I_ss, axis=0), bin_size=2)
        ax[2,1].semilogy(rad, intens, color='b')
        intens, rad = rad_average(np.sum(I_ps, axis=0), bin_size=2)
        ax[2,1].semilogy(rad, intens, color='g')
        intens, rad = rad_average(np.sum(I_nufft, axis=0), bin_size=2)
        ax[2,1].semilogy(rad, intens, color='g')
        """
    for i in range(0,I_ss.shape[0]):
        print(i+1)
        #im = ax[4,0].imshow(np.sum(I_nufft, axis=0), cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
        im = ax[0,i+1].imshow(I_ss[i], cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
        #fig.colorbar(im, ax=ax[0,i+1], orientation='vertical', pad=0.01, fraction=0.048)
        #ax[2,i+1].set_title("SS level %d" %i)
        
        im = ax[1,i+1].imshow(I_ps[i], cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
        #fig.colorbar(im, ax=ax[1,i+1], orientation='vertical', pad=0.01, fraction=0.048)
        #ax[2,i+1].set_title("PS level %d" %i)

        im = ax[2,i+1].imshow(I_nufft[i], cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())

        if(psf):
            intens, rad = rad_average(I_ss[i], bin_size=2)
            ax[3,i+1].semilogy(rad, intens, color='b')
            intens, rad = rad_average(I_ps[i], bin_size=2)
            ax[3,i+1].semilogy(rad, intens, color='g')
            intens, rad = rad_average(I_nufft[i], bin_size=2)
            ax[3,i+1].semilogy(rad, intens, color='r', ls='--')
    plt.subplots_adjust(wspace=0.5)


def Plot_PSF_profile(I_ss, I_nufft):
    print('Plot PSF profile')
    fig, ax = plt.subplots(ncols=2, nrows = 2, figsize=(10, 8))
    ax = ax.flatten()
    N_pix = I_ss.shape[-1]
    my_ext = [-N_pix//2, N_pix//2, -N_pix//2, N_pix//2]
    im = ax[0].imshow(I_ss, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[0], orientation='vertical', pad=0.01, fraction=0.048)
    ax[0].set_title('Interpolated SS')

    im = ax[1].imshow(I_nufft, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[1], orientation='vertical', pad=0.01, fraction=0.048)
    ax[1].set_title('NUFFT')

    intens, rad = rad_average(I_ss, bin_size=2)
    ax[2].semilogy(rad, intens, color='b', label='SS')
    intens, rad = rad_average(I_nufft, bin_size=2)
    ax[2].semilogy(rad, intens, color='r', ls='--', label='NUFFT')
    ax[2].legend()


def Plot_Comparison(I_ss, I_nufft):
    print('Plot Comparison')
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))
    ax = ax.flatten()
    N_pix = I_ss.shape[-1]
    my_ext = [-N_pix//2, N_pix//2, -N_pix//2, N_pix//2]
    
    I_ss = np.sum(I_ss, axis=0)
    I_nufft = np.sum(I_nufft, axis=0)

    im = ax[0].imshow(I_ss, cmap='cubehelix', origin='lower', extent=my_ext)#, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[0], orientation='vertical', pad=0.01, fraction=0.048)
    ax[0].set_title('Interpolated SS')

    im = ax[1].imshow(I_nufft, cmap='cubehelix', origin='lower', extent=my_ext)#, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[1], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[2].scatter(0, 0, s=512, facecolors='none', edgecolors='r')
    ax[1].set_title('NUFFT')

    im = ax[2].imshow(I_ss/I_nufft, cmap='cubehelix', origin='lower', extent=my_ext)#, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[3], orientation='vertical', pad=0.01, fraction=0.048)
    ax[2].set_title('SS/NUFFT')


    for a in ax[:-1]:
        a.axes.get_yaxis().set_visible(False)
        a.axes.get_xaxis().set_visible(False)


def Plot_Levels(I_ss, I_nufft, D_ss, D_nufft):
    print('Plot Levels')
    N_pix = I_ss.shape[-1]
    my_ext = [-N_pix//2, N_pix//2, -N_pix//2, N_pix//2]

    print(I_ss.shape, I_ss.shape[0])

    I_ss_sum = np.sum(I_ss, axis=0)
    I_nufft_sum = np.sum(I_nufft, axis=0)
    """
    I_ss_substr = I_ss[0]
    for i in range(1,I_ss.shape[0]):
        I_ss_substr -= I_ss[i]
    I_ps_substr = I_ps[0]
    for i in range(1,I_ps.shape[0]):
        I_ps_substr -= I_ps[i]
    I_nufft_substr = I_ps[0]
    for i in range(1,I_nufft.shape[0]):
        I_nufft_substr -= I_nufft[i]
    """

    fig, ax = plt.subplots(ncols=I_ss.shape[0]+2, nrows=3, figsize=(12, 8))

    im = ax[0,0].imshow(I_ss_sum, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[0,0], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[0,0].set_title('Interpolated SS')

    #im = ax[1,0].imshow(I_ps_sum, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    #fig.colorbar(im, ax=ax[1,0], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[1,0].set_title('Interpolated PS')

    im = ax[1,0].imshow(I_nufft_sum, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[1,0], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[2].scatter(0, 0, s=512, facecolors='none', edgecolors='r')
    #ax[2,0].set_title('NUFFT')

    norm_I_ss = np.sum(I_ss*D_ss, axis=0)
    #norm_I_ps = np.sum(RescaleData(I_ps, a=0, b=1), axis=0)
    norm_I_nufft = np.sum(I_nufft*D_nufft, axis=0)

    im = ax[0,1].imshow(norm_I_ss, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[0,1], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[0,0].set_title('Interpolated SS')

    #im = ax[1,0].imshow(norm_I_ps, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    #fig.colorbar(im, ax=ax[1,0], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[1,0].set_title('Interpolated PS')

    im = ax[1,1].imshow(norm_I_nufft, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[1,1], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[2].scatter(0, 0, s=512, facecolors='none', edgecolors='r')
    #ax[2,0].set_title('NUFFT')

    # PSF radial distribution profile
    intens, rad = rad_average(norm_I_ss, bin_size=2)
    ax[2,1].semilogy(rad, intens, color='b', label='SS')
    #intens, rad = rad_average(I_ps_sum, bin_size=2)
    #ax[3,0].semilogy(rad, intens, color='g', label='PS')
    intens, rad = rad_average(norm_I_nufft, bin_size=2)
    ax[2,1].semilogy(rad, intens, color='r', ls='--', label='NUFFT')

    # PSF radial distribution profile
    intens, rad = rad_average(I_ss_sum, bin_size=2)
    ax[2,0].semilogy(rad, intens, color='b', label='SS')
    #intens, rad = rad_average(I_ps_sum, bin_size=2)
    #ax[3,0].semilogy(rad, intens, color='g', label='PS')
    intens, rad = rad_average(I_nufft_sum, bin_size=2)
    ax[2,0].semilogy(rad, intens, color='r', ls='--', label='NUFFT')
    ax[2,0].legend()

    for i in range(0, I_ss.shape[0]):
        # Eigen Level
        im = ax[0,i+2].imshow(I_ss[i], cmap='Blues_r', origin='lower', extent=my_ext, norm=colors.LogNorm())
        #im = ax[1,i+2].imshow(I_ps[i], cmap='Blues_r', origin='lower', extent=my_ext, norm=colors.LogNorm())
        im = ax[1,i+2].imshow(I_nufft[i], cmap='Blues_r', origin='lower', extent=my_ext, norm=colors.LogNorm())

        # 
        #im = ax[0,i].imshow(I_ss, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
        #fig.colorbar(im, ax=ax[3], orientation='vertical', pad=0.01, fraction=0.048)
        #ax[3].set_title('SS/NUFFT')

        #im = ax[4].imshow(I_ps/I_nufft, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
        #fig.colorbar(im, ax=ax[4], orientation='vertical', pad=0.01, fraction=0.048)
        #ax[4].set_title('PS/NUFFT')

        # PSF radial distribution profile
        intens, rad = rad_average(I_ss[i], bin_size=2)
        ax[2,i+2].semilogy(rad, intens, color='b', label='SS')
        #intens, rad = rad_average(I_ps_sum, bin_size=2)
        #ax[3,i+2].semilogy(rad, i  ntens, color='g', label='PS')
        intens, rad = rad_average(I_nufft[i], bin_size=2)
        ax[2,i+2].semilogy(rad, intens, color='r', ls='--', label='NUFFT')

    #fig.delaxes(ax[5])
    for a in ax.flatten():
        a.axes.get_yaxis().set_visible(False)
        a.axes.get_xaxis().set_visible(False)
    
    #a.axes.get_yaxis().set_visible(False)
    #a.axes.get_xaxis().set_visible(False)


def Plot_Merge_Levels(sky_mod, I_ss, I_ps, I_nufft, D_ss, D_nufft):
    print('Plot Levels')
    N_pix = I_ss.shape[-1]
    my_ext = [-N_pix//2, N_pix//2, -N_pix//2, N_pix//2]

    print(I_ss.shape, I_ss.shape[0])

    I_ss_sum = np.sum(I_ss, axis=0)
    I_ps_sum = np.sum(I_ps, axis=0)
    I_nufft_sum = np.sum(I_nufft, axis=0)
    """
    I_ss_substr = I_ss[0]
    for i in range(1,I_ss.shape[0]):
        I_ss_substr -= I_ss[i]
    I_ps_substr = I_ps[0]
    for i in range(1,I_ps.shape[0]):
        I_ps_substr -= I_ps[i]
    I_nufft_substr = I_ps[0]
    for i in range(1,I_nufft.shape[0]):
        I_nufft_substr -= I_nufft[i]
    """

    fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(12, 8))

    im = ax[0,0].imshow(I_ss_sum, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[0,0], orientation='vertical', pad=0.01, fraction=0.048)
    ax[0,0].scatter(*sky_mod, facecolors="none", edgecolors="w")
    #ax[0,0].set_title('Interpolated SS')


    im = ax[1,0].imshow(I_nufft_sum, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[1,0], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[2].scatter(0, 0, s=512, facecolors='none', edgecolors='r')
    #ax[2,0].set_title('NUFFT')

    norm_I_ss = np.sum(I_ss*D_ss, axis=0)
    #norm_I_ps = np.sum(RescaleData(I_ps, a=0, b=1), axis=0)
    norm_I_nufft = np.sum(I_nufft*D_nufft, axis=0)

    im = ax[0,1].imshow(norm_I_ss, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[0,1], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[0,0].set_title('Interpolated SS')

    im = ax[1,1].imshow(norm_I_nufft, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[1,1], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[2].scatter(0, 0, s=512, facecolors='none', edgecolors='r')
    #ax[2,0].set_title('NUFFT')

    # PSF radial distribution profile
    intens, rad = rad_average(norm_I_ss, bin_size=2)
    ax[2,1].semilogy(rad, intens, color='b', label='SS')
    intens, rad = rad_average(norm_I_nufft, bin_size=2)
    ax[2,1].semilogy(rad, intens, color='r', ls='--', label='NUFFT')

    # PSF radial distribution profile
    intens, rad = rad_average(I_ss_sum, bin_size=2)
    ax[2,0].semilogy(rad, intens, color='b', label='SS')
    intens, rad = rad_average(I_nufft_sum, bin_size=2)
    ax[2,0].semilogy(rad, intens, color='r', ls='--', label='NUFFT')
    ax[2,0].legend()

    # Eigen Level
    im = ax[0,2].imshow(I_ss[0], cmap='Blues_r', origin='lower', extent=my_ext, norm=colors.LogNorm())
    im = ax[1,2].imshow(I_nufft[0], cmap='Blues_r', origin='lower', extent=my_ext, norm=colors.LogNorm())

    # 
    #im = ax[0,i].imshow(I_ss, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    #fig.colorbar(im, ax=ax[3], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[3].set_title('SS/NUFFT')

    # PSF radial distribution profile
    intens, rad = rad_average(I_ss[0], bin_size=2)
    ax[2,2].semilogy(rad, intens, color='b', label='SS')
    intens, rad = rad_average(I_nufft[0], bin_size=2)
    ax[2,2].semilogy(rad, intens, color='r', ls='--', label='NUFFT')

    # Higher Eigen Level
    im = ax[0,3].imshow(np.sum(I_ss[1:], axis=0), cmap='Blues_r', origin='lower', extent=my_ext, norm=colors.LogNorm())
    im = ax[1,3].imshow(np.sum(I_nufft[1:], axis=0), cmap='Blues_r', origin='lower', extent=my_ext, norm=colors.LogNorm())


    # PSF radial distribution profile
    intens, rad = rad_average(np.sum(I_ss[1:], axis=0), bin_size=2)
    ax[2,3].semilogy(rad, intens, color='b', label='SS')

    intens, rad = rad_average(np.sum(I_nufft[1:], axis=0), bin_size=2)
    ax[2,3].semilogy(rad, intens, color='r', ls='--', label='NUFFT')

    for a in ax.flatten():
        a.axes.get_yaxis().set_visible(False)
        a.axes.get_xaxis().set_visible(False)



def Plot_Complete(I_clean, I_dirty, I_ss, I_nufft, D_ss, D_nufft):
    print('Plot Complete')

    image_data = fits.getdata(I_clean, ext=0)
    I_clean = np.fliplr(np.squeeze(image_data))

    image_data = fits.getdata(I_dirty, ext=0)
    I_dirty = np.fliplr(np.squeeze(image_data))

    I_clean = RescaleData(I_clean, a=0, b=I_clean.max())
    print(I_dirty.min(), I_dirty.max())
    I_dirty = RescaleData(I_dirty, a=0, b=I_dirty.max())
    #I_ss = RescaleData(I_ss, a=0, b=I_ss.max())
    #I_nufft = RescaleData(I_nufft, a=0, b=I_nufft.max())

    N_pix = I_ss.shape[-1]
    my_ext = [-N_pix//2, N_pix//2, -N_pix//2, N_pix//2]

    norm_I_ss = np.sum(I_ss*D_ss, axis=0)
    norm_I_nufft = np.sum(I_nufft*D_nufft, axis=0)

    fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(24, 16))

    # Plot ground truth image
    im = ax[0,0].imshow(I_clean, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[0,0], pad=0.01, fraction=0.048)
    ax[0,0].set_title('True')

    # Plot dirty image
    im = ax[0,1].imshow(I_dirty, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[0,1], orientation='vertical', pad=0.01, fraction=0.048)
    ax[0,1].set_title('Dirty')

    # Plot Bluebild image, sum of all eigenlevels with eigenvalues normalisation
    im = ax[0,2].imshow(norm_I_ss, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[0,2], orientation='vertical', pad=0.01, fraction=0.048)
    ax[0,2].set_title('SS')

    im = ax[0,3].imshow(norm_I_nufft, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax[0,3], orientation='vertical', pad=0.01, fraction=0.048)
    ax[0,3].set_title('NUFFT')

    #im = ax[0,4].imshow(I_wsclean, cmap='cubehelix', origin='lower', extent=my_ext, norm=colors.LogNorm())
    #fig.colorbar(im, ax=ax[0,4], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[0,4].set_title('WSCLEAN')

    for i in range(0, I_ss.shape[0]):
        # Eigen Level
        im = ax[1,i].imshow(I_ss[i]*D_ss[i], cmap='Blues_r', origin='lower', extent=my_ext, norm=colors.LogNorm())
        im = ax[2,i].imshow(I_nufft[i]*D_nufft[i], cmap='Blues_r', origin='lower', extent=my_ext, norm=colors.LogNorm())
        if(i == 0):
            ax[1,i].set_title('SS: E=%d' %i)
            ax[2,i].set_title('NUFFT: E=%d' %i)
        else:
            ax[1,i].set_title('E=%d' %i)
            ax[2,i].set_title('E=%d' %i)
        
        """
        # PSF radial distribution profile
        intens, rad = rad_average(I_ss[i], bin_size=2)
        ax[2,i+2].semilogy(rad, intens, color='b', label='SS')
        intens, rad = rad_average(I_nufft[i], bin_size=2)
        ax[2,i+2].semilogy(rad, intens, color='r', ls='--', label='NUFFT')
        """

    for a in ax.flatten():
        a.axes.get_yaxis().set_visible(False)
        a.axes.get_xaxis().set_visible(False)
    
    #a.axes.get_yaxis().set_visible(False)
    #a.axes.get_xaxis().set_visible(False)


#============================================================================================

#path = '/users/mibianco/data/user_catalog/'
path = '/users/mibianco/data/psf/'
#path = '/users/mibianco/data/lofar/'

cname = 'BL_Nlvl%d_Nsrc%d' %(4, 1)
#cname = 'lofar30MHz153'
I_lsq_eq_ss_interp_data = np.load('%sI_ss_%s.npy' %(path, cname))
I_lsq_eq_nufft_data = np.load('%sI_nufft_%s.npy' %(path, cname))
D_eig = np.load('%sD_%s.npy' %(path, cname))
print(I_lsq_eq_nufft_data.shape, I_lsq_eq_ss_interp_data.shape, D_eig.shape)

#I_lsq_eq_ss_interp_data = np.load('%sI_lsq_eq_ss_interp_Nsrc%d_Nlvl%d.npy' %(path, N_src, N_level))    # first level only
#I_lsq_eq_nufft_data = np.load('%sI_lsq_eq_nufft_Nsrc%d_Nlvl%d.npy' %(path, N_src, N_level))

#norm_I_lsq_eq_ss_interp_data = RescaleData(I_lsq_eq_ss_interp_data, a=0, b=1)
#norm_I_lsq_eq_nufft_data = RescaleData(I_lsq_eq_nufft_data, a=0, b=1)


# Plot results ==========================================================================
#Plot_PSF_profile(I_ss=I_lsq_eq_ss_interp_data, I_nufft=I_lsq_eq_nufft_data)
#plt.savefig("%spsf" %path, bbox_inches='tight')

#Plot_PSF_profile(I_ss=norm_I_lsq_eq_ss_interp_data, I_nufft=norm_I_lsq_eq_nufft_data)
#plt.savefig("%stest_normPSFprofile" %path, bbox_inches='tight')

#Plot_Comparison(I_ss=I_lsq_eq_ss_interp_data, I_nufft=I_lsq_eq_nufft_data)
#plt.savefig("%stest_compare" %path, bbox_inches='tight')

#Plot_Comparison(I_ss=norm_I_lsq_eq_ss_interp_data, I_ps=norm_I_lsq_eq_ps_interp_data, I_nufft=norm_I_lsq_eq_nufft_data)
#plt.savefig("%stest_normcompare" %path, bbox_inches='tight')

#draw_levels(I_ss=I_lsq_eq_ss_interp_data, I_ps=I_lsq_eq_ps_interp_data, I_nufft=I_lsq_eq_nufft_data)
#plt.savefig("%stest_Nsrc%d_Nlvl%d.png" %(path, N_src, N_level), bbox_inches='tight')

#Plot_Complete(I_clean=path+'RADIO30MHz153.fits', I_dirty=path+'lofar30MHz153-dirty.fits', I_ss=I_lsq_eq_ss_interp_data, I_nufft=I_lsq_eq_nufft_data, D_ss=D_eig, D_nufft=D_eig)
#plt.savefig("%s_complete.png" %(path+cname), bbox_inches='tight')

Plot_Levels(I_ss=I_lsq_eq_ss_interp_data, I_nufft=I_lsq_eq_nufft_data, D_ss=D_eig, D_nufft=D_eig)
plt.savefig("%slevels_%s.png" %(path, cname), bbox_inches='tight')

#mock_catalog = np.array([[216.9, 32.8, 1e6], [218.2, 34.8, 1e6], [218.8, 32.8, 1e6], [217.8, 32.4, 1e6]]) 
#Plot_Merge_Levels(sky_mod=mock_catalog, I_ss=I_lsq_eq_ss_interp_data, I_ps=I_lsq_eq_ps_interp_data, I_nufft=I_lsq_eq_nufft_data, D_ss=D_ss, D_nufft=D_nufft)
#plt.savefig("%stest_merge_Nsrc%d_Nlvl%d.png" %(path, N_src, N_level), bbox_inches='tight')

#for a in ax[:-1]:
    #a.axes.get_yaxis().set_visible(False)
    #a.axes.get_xaxis().set_visible(False)
#    pass
#plt.savefig("%stest_bluebild_PSF_Nsrc%d_Nlvl%d" %(path, N_src, N_level), bbox_inches='tight')
#plt.savefig("%stest_bluebild_PSF_test" %path, bbox_inches='tight')
#plt.savefig("%stest_bluebild_PSF_Nsrc%d_Nlvl%d_rescale" %(path, N_src, N_level), bbox_inches='tight')
