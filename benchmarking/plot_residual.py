import matplotlib as mpl
mpl.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from other_utils import rad_average, RescaleData

path = '/users/mibianco/data/test_PSF/'

I_ss_full, I_nufft_full = np.load('%sI_ss_full.npy' %path), np.load('%sI_nufft_full.npy' %path) 
D_full = np.load('%sD_full.npy' %path) 
I_ss, I_nufft = D_full*I_ss_full, D_full*I_nufft_full 
#I_ss, I_nufft = I_ss_full, I_nufft_full 
 

I_ss_res = np.zeros_like(I_ss) 
I_nufft_res = np.zeros_like(I_ss)  
 
for i, typ in enumerate(['BL', 'TL', 'BR', 'TR']): 
    D_type = np.load('%sD_%s.npy' %(path, typ)) 
     
    I_ss_typ = np.load('%sI_ss_%s.npy' %(path, typ)) * D_type 
    I_ss_res[i] = 1- I_ss[i] / I_ss_typ #/I_ss[i] 
     
    I_nufft_typ = np.load('%sI_nufft_%s.npy' %(path, typ)) * D_type 
    I_nufft_res[i] = 1-I_nufft[i] / I_nufft_typ  # )/I_nufft[i] 
#np.save('I_ss_res.npy', I_ss_res) 
#np.save('I_nufft_res.npy', I_nufft_res) 

"""
I_ss, I_nufft = np.load(path+'I_ss_full.npy'), np.load(path+'I_nufft_full.npy')
I_ss_res, I_nufft_res = RescaleData(np.load(path+'I_ss_res.npy'), a=0.001, b=1), RescaleData(np.load(path+'I_nufft_res.npy'), a=0.001, b=1)
D_ss = np.load(path+'D_full.npy')
D_nufft = D_ss
"""

print('Plot Levels')
N_pix = I_ss.shape[-1]
my_ext = [-N_pix//2, N_pix//2, -N_pix//2, N_pix//2]

print(I_ss.shape, I_ss.shape[0])

I_ss_sum = np.sum(I_ss_full, axis=0)
I_nufft_sum = np.sum(I_nufft_full, axis=0)

fig, ax = plt.subplots(ncols=I_ss.shape[0]+2, nrows=3, figsize=(12, 8))

# First column
im = ax[0,0].imshow(I_ss_sum, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
fig.colorbar(im, ax=ax[0,0], orientation='vertical', pad=0.01, fraction=0.048)
ax[0,0].set_title('Interpolated SS')

im = ax[1,0].imshow(I_nufft_sum, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
fig.colorbar(im, ax=ax[1,0], orientation='vertical', pad=0.01, fraction=0.048)
ax[1,0].set_title('NUFFT')

# Second column
norm_I_ss = np.sum(I_ss, axis=0)
norm_I_nufft = np.sum(I_nufft, axis=0)

im = ax[0,1].imshow(norm_I_ss, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
fig.colorbar(im, ax=ax[0,1], orientation='vertical', pad=0.01, fraction=0.048)
#ax[0,0].set_title('Interpolated SS')

im = ax[1,1].imshow(norm_I_nufft, cmap='cubehelix', origin='lower', norm=colors.LogNorm(), extent=my_ext)
fig.colorbar(im, ax=ax[1,1], orientation='vertical', pad=0.01, fraction=0.048)
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

for i in range(0, I_ss.shape[0]):
    # Eigen Level
    print(I_ss_res[i].min(), I_ss_res[i].max())
    im = ax[0,i+2].imshow(I_ss_res[i], cmap='tab10', origin='lower', norm=colors.LogNorm(vmin=0.5, vmax=1), extent=my_ext)
    fig.colorbar(im, ax=ax[0,i+2], orientation='vertical', pad=0.01, fraction=0.048)
    im = ax[1,i+2].imshow(I_nufft_res[i], cmap='tab10', origin='lower', norm=colors.LogNorm(vmin=0.5, vmax=1), extent=my_ext)
    fig.colorbar(im, ax=ax[1,i+2], orientation='vertical', pad=0.01, fraction=0.048)

    # PSF radial distribution profile
    intens, rad = rad_average(I_ss_res[i], bin_size=2)
    ax[2,i+2].semilogy(rad, intens, color='b', label='SS')
    #intens, rad = rad_average(I_ps_sum, bin_size=2)
    #ax[3,i+2].semilogy(rad, i  ntens, color='g', label='PS')
    intens, rad = rad_average(I_nufft_res[i], bin_size=2)
    ax[2,i+2].semilogy(rad, intens, color='r', ls='--', label='NUFFT')

#fig.delaxes(ax[5])
for a in ax.flatten():
    a.axes.get_yaxis().set_visible(False)
    a.axes.get_xaxis().set_visible(False)

#a.axes.get_yaxis().set_visible(False)
#a.axes.get_xaxis().set_visible(False)
plt.savefig("%slevels_%s.png" %(path, 'res'), bbox_inches='tight')
