import matplotlib as mpl
mpl.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from other_utils import rad_average, RescaleData

def draw_levels(I_ss, I_ps, I_nufft, psf=True):
    if(psf):
        fig, ax = plt.subplots(ncols=I_ss.shape[0]+1, nrows=4, figsize=(15, 8))
    else:
        fig, ax = plt.subplots(ncols=I_ss.shape[0]+1, nrows=3, figsize=(15, 5))
    fig.tight_layout(pad = 2.0)
    
    my_ext = [-I_ss.shape[-1]//2, I_ss.shape[-1]//2, -I_ss.shape[-1]//2, I_ss.shape[-1]//2]

    ax[0,0].set_title("SS")
    im = ax[0,0].imshow(np.sum(I_ss, axis=0), cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[0,0], orientation='vertical', pad=0.01, fraction=0.048)

    ax[1,0].set_title("PS")
    im = ax[1,0].imshow(np.sum(I_ps, axis=0), cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[1,0], orientation='vertical', pad=0.01, fraction=0.048)

    ax[2,0].set_title("NUFFT")
    im = ax[2,0].imshow(np.sum(I_nufft, axis=0), cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
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
        #im = ax[4,0].imshow(np.sum(I_nufft, axis=0), cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
        im = ax[0,i+1].imshow(I_ss[i], cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
        #fig.colorbar(im, ax=ax[0,i+1], orientation='vertical', pad=0.01, fraction=0.048)
        #ax[2,i+1].set_title("SS level %d" %i)
        
        im = ax[1,i+1].imshow(I_ps[i], cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
        #fig.colorbar(im, ax=ax[1,i+1], orientation='vertical', pad=0.01, fraction=0.048)
        #ax[2,i+1].set_title("PS level %d" %i)

        im = ax[2,i+1].imshow(I_nufft[i], cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)

        if(psf):
            intens, rad = rad_average(I_ss[i], bin_size=2)
            ax[3,i+1].semilogy(rad, intens, color='b')
            intens, rad = rad_average(I_ps[i], bin_size=2)
            ax[3,i+1].semilogy(rad, intens, color='g')
            intens, rad = rad_average(I_nufft[i], bin_size=2)
            ax[3,i+1].semilogy(rad, intens, color='r', ls='--')
    plt.subplots_adjust(wspace=0.5)


def Plot_PSF_profile(I_ss, I_ps, I_nufft):
    print('Plot PSF profile')
    fig, ax = plt.subplots(ncols=2, nrows = 2, figsize=(10, 8))
    ax = ax.flatten()
    N_pix = I_ss.shape[-1]
    my_ext = [-N_pix//2, N_pix//2, -N_pix//2, N_pix//2]
    im = ax[0].imshow(I_ss, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[0], orientation='vertical', pad=0.01, fraction=0.048)
    ax[0].set_title('Interpolated SS')

    im = ax[1].imshow(I_ps, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[1], orientation='vertical', pad=0.01, fraction=0.048)
    ax[1].set_title('Interpolated PS')

    im = ax[2].imshow(I_nufft, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[2], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[2].scatter(0, 0, s=512, facecolors='none', edgecolors='r')
    ax[2].set_title('NUFFT')

    intens, rad = rad_average(I_ss, bin_size=2)
    ax[3].semilogy(rad, intens, color='b', label='SS')
    intens, rad = rad_average(I_ps, bin_size=2)
    ax[3].semilogy(rad, intens, color='g', label='PS')
    intens, rad = rad_average(I_nufft, bin_size=2)
    ax[3].semilogy(rad, intens, color='r', ls='--', label='NUFFT')
    ax[3].legend()


def Plot_Comparison(I_ss, I_ps, I_nufft):
    print('Plot Comparison')
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 8))
    ax = ax.flatten()
    N_pix = I_ss.shape[-1]
    my_ext = [-N_pix//2, N_pix//2, -N_pix//2, N_pix//2]

    im = ax[0].imshow(I_ss, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[0], orientation='vertical', pad=0.01, fraction=0.048)
    ax[0].set_title('Interpolated SS')

    im = ax[1].imshow(I_ps, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[1], orientation='vertical', pad=0.01, fraction=0.048)
    ax[1].set_title('Interpolated PS')

    im = ax[2].imshow(I_nufft, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[2], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[2].scatter(0, 0, s=512, facecolors='none', edgecolors='r')
    ax[2].set_title('NUFFT')

    im = ax[3].imshow(I_ss/I_nufft, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[3], orientation='vertical', pad=0.01, fraction=0.048)
    ax[3].set_title('SS/NUFFT')

    im = ax[4].imshow(I_ps/I_nufft, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[4], orientation='vertical', pad=0.01, fraction=0.048)
    ax[4].set_title('PS/NUFFT')

    fig.delaxes(ax[5])
    for a in ax[:-1]:
        a.axes.get_yaxis().set_visible(False)
        a.axes.get_xaxis().set_visible(False)


def Plot_Levels(I_ss, I_ps, I_nufft):
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

    fig, ax = plt.subplots(ncols=I_ss.shape[0], nrows=4, figsize=(10, 8))

    im = ax[0,0].imshow(I_ss_sum, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[0,0], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[0,0].set_title('Interpolated SS')

    im = ax[1,0].imshow(I_ps_sum, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[1,0], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[1,0].set_title('Interpolated PS')

    im = ax[2,0].imshow(I_nufft_sum, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
    fig.colorbar(im, ax=ax[2,0], orientation='vertical', pad=0.01, fraction=0.048)
    #ax[2].scatter(0, 0, s=512, facecolors='none', edgecolors='r')
    #ax[2,0].set_title('NUFFT')

    # PSF radial distribution profile
    intens, rad = rad_average(I_ss_sum, bin_size=2)
    ax[3,0].semilogy(rad, intens, color='b', label='SS')
    intens, rad = rad_average(I_ps_sum, bin_size=2)
    ax[3,0].semilogy(rad, intens, color='g', label='PS')
    intens, rad = rad_average(I_nufft_sum, bin_size=2)
    ax[3,0].semilogy(rad, intens, color='r', label='NUFFT')
    ax[3,0].legend()

    for i in range(0, I_ss.shape[0]):
        # Eigen Level
        im = ax[0,i+2].imshow(I_ss[i], cmap='Blues_r', origin='lower', norm=colors.LogNorm(), extent=my_ext)
        im = ax[1,i+2].imshow(I_ps[i], cmap='Blues_r', origin='lower', norm=colors.LogNorm(), extent=my_ext)
        im = ax[2,i+2].imshow(I_nufft[i], cmap='Blues_r', origin='lower', norm=colors.LogNorm(), extent=my_ext)

        # 
        #im = ax[0,i].imshow(I_ss, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
        #fig.colorbar(im, ax=ax[3], orientation='vertical', pad=0.01, fraction=0.048)
        #ax[3].set_title('SS/NUFFT')

        #im = ax[4].imshow(I_ps/I_nufft, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
        #fig.colorbar(im, ax=ax[4], orientation='vertical', pad=0.01, fraction=0.048)
        #ax[4].set_title('PS/NUFFT')

        # PSF radial distribution profile
        intens, rad = rad_average(I_ss_sum, bin_size=2)
        ax[3,i+2].semilogy(rad, intens, color='b', label='SS')
        intens, rad = rad_average(I_ps_sum, bin_size=2)
        ax[3,i+2].semilogy(rad, intens, color='g', label='PS')
        intens, rad = rad_average(I_nufft_sum, bin_size=2)
        ax[3,i+2].semilogy(rad, intens, color='r', label='NUFFT')

    #fig.delaxes(ax[5])
    for a in ax.flatten():
        a.axes.get_yaxis().set_visible(False)
        a.axes.get_xaxis().set_visible(False)

#============================================================================================

path = '/users/mibianco/data/user_catalog/'
N_src = 1
N_level = 4

I_lsq_eq_ss_interp_data = np.load('%sI_lsq_eq_ss_interp_Nsrc%d_Nlvl%d.npy' %(path, N_src, N_level))    # first level only
I_lsq_eq_ps_interp_data = np.load('%sI_lsq_eq_ps_interp_Nsrc%d_Nlvl%d.npy' %(path, N_src, N_level))
I_lsq_eq_nufft_data = np.load('%sI_lsq_eq_nufft_Nsrc%d_Nlvl%d.npy' %(path, N_src, N_level))
print(I_lsq_eq_nufft_data.shape)

norm_I_lsq_eq_ss_interp_data = RescaleData(I_lsq_eq_ss_interp_data, a=0, b=1)
norm_I_lsq_eq_ps_interp_data = RescaleData(I_lsq_eq_ps_interp_data, a=0, b=1)
norm_I_lsq_eq_nufft_data = RescaleData(I_lsq_eq_nufft_data, a=0, b=1)


# Plot results ==========================================================================
Plot_PSF_profile(I_ss=I_lsq_eq_ss_interp_data, I_ps=I_lsq_eq_ps_interp_data, I_nufft=I_lsq_eq_nufft_data)
plt.savefig("%stest_PSFprofile" %path, bbox_inches='tight')

Plot_PSF_profile(I_ss=norm_I_lsq_eq_ss_interp_data, I_ps=norm_I_lsq_eq_ps_interp_data, I_nufft=norm_I_lsq_eq_nufft_data)
plt.savefig("%stest_normPSFprofile" %path, bbox_inches='tight')

Plot_Comparison(I_ss=I_lsq_eq_ss_interp_data, I_ps=I_lsq_eq_ps_interp_data, I_nufft=I_lsq_eq_nufft_data)
plt.savefig("%stest_compare" %path, bbox_inches='tight')

Plot_Comparison(I_ss=norm_I_lsq_eq_ss_interp_data, I_ps=norm_I_lsq_eq_ps_interp_data, I_nufft=norm_I_lsq_eq_nufft_data)
plt.savefig("%stest_normcompare" %path, bbox_inches='tight')

#draw_levels(I_ss=I_lsq_eq_ss_interp_data, I_ps=I_lsq_eq_ps_interp_data, I_nufft=I_lsq_eq_nufft_data)
#plt.savefig("%stest_Nsrc%d_Nlvl%d.png" %(path, N_src, N_level), bbox_inches='tight')

Plot_Levels(I_ss=I_lsq_eq_ss_interp_data, I_ps=I_lsq_eq_ps_interp_data, I_nufft=I_lsq_eq_nufft_data)
plt.savefig("%stest_Nsrc%d_Nlvl%d.png" %(path, N_src, N_level), bbox_inches='tight')


#for a in ax[:-1]:
    #a.axes.get_yaxis().set_visible(False)
    #a.axes.get_xaxis().set_visible(False)
#    pass
#plt.savefig("%stest_bluebild_PSF_Nsrc%d_Nlvl%d" %(path, N_src, N_level), bbox_inches='tight')
#plt.savefig("%stest_bluebild_PSF_test" %path, bbox_inches='tight')
#plt.savefig("%stest_bluebild_PSF_Nsrc%d_Nlvl%d_rescale" %(path, N_src, N_level), bbox_inches='tight')
