from astropy.io import fits
import astropy.wcs as pywcs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import numpy as np

with fits.open("bluebild_processed_4gauss.fits") as hdul:
    print(hdul)
    grid = hdul[0].data
    data = hdul[1].data

with fits.open("/home/etolley/data/gauss4/gauss4-image-pb.fits") as hdul:
    wcs = pywcs.WCS(hdul[0].header)
    clean_data = np.flip(hdul[0].data[0,0,:,:])

print(wcs)
corner_coords = wcs.wcs_pix2world([[ 0,0,0,0], [ 2000,2000,0,0]], 0)
corners = [(c[0],c[1]) for c in corner_coords]
a1, b1 = corners[0][0], corners[0][1]
a2, b2 = corners[1][0], corners[1][1]

x1, y1 = grid[0,0,0], grid[1,0,0]
x2, y2 = grid[0,-1,-1], grid[1,-1,-1]
print("coord diffs:", x2-x1, y2-y1, a2-a1, b2-b1)

with fits.open("/home/etolley/data/gauss4/C_4gaussian-model.fits") as hdul:
    true_data = np.flip(hdul[0].data[:,:])
 
print(clean_data.shape)



fig, axs = plt.subplots(2, 3)
#fig.tight_layout(pad = 0.5)
plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.2, hspace = 0.3)

import copy
my_cmap = copy.copy(cm.get_cmap("GnBu_r")) # copy the default cmap
my_cmap.set_bad((0,0,0))

bb_data = np.sum(data,axis = 0)

bb_extent = [-400,2400,-800,2800]

source_locs = [(180, 530), (460, 1550), (1360, 1367), (1439, 20)]

def rect(s):
    return plt.Rectangle(s, 300, 300, fill = False, edgecolor="r") 

axs[0,0].imshow(bb_data, cmap = my_cmap, norm=LogNorm(vmin=0.001, vmax=np.max(bb_data)),extent= bb_extent, origin='lower')
axs[0,0].set_title("BB Processed image")
#for s in source_locs: axs[0,0].add_patch(rect(s))

axs[0,1].imshow(grid[0,:,:], cmap = my_cmap, origin='lower', )
axs[0,1].set_title("BB Fits grid[0]")
axs[0,2].imshow(grid[1,:,:], cmap = my_cmap,origin='lower',)
axs[0,2].set_title("BB Fits grid[1]")
axs[1,0].imshow(true_data, cmap = my_cmap, norm=LogNorm(vmin=0.00001, vmax=np.max(true_data)), origin='lower' )
axs[1,0].set_title("True image (flipped)")
#for s in source_locs: axs[1,0].add_patch(rect(s))
axs[1,1].imshow(clean_data, cmap = my_cmap, norm=LogNorm(vmin=0.01, vmax=np.max(clean_data)), origin='lower' )
axs[1,1].set_title("WSClean image (flipped)")
#for s in source_locs: axs[1,1].add_patch(rect(s))
#axs[1,2].imshow(bb_data, cmap = my_cmap, norm=LogNorm(vmin=0.001, vmax=np.max(bb_data)),origin='lower', extent= bb_extent)
#axs[1,2].imshow(true_data, cmap = my_cmap, norm=LogNorm(vmin=0.00001, vmax=np.max(true_data)), alpha=0.5, origin='lower' )

true_fluxes, clean_fluxes, bb_fluxes = [],[],[]
for s in source_locs:
    axs[0,0].add_patch(rect(s))
    axs[1,0].add_patch(rect(s))
    axs[1,1].add_patch(rect(s))
    true_flux  = np.sum(true_data[ s[1]:s[1]+300, s[0]:s[0]+300])
    clean_flux = np.sum(clean_data[s[1]:s[1]+300, s[0]:s[0]+300])
    bb_x_start = int((s[1] - bb_extent[2])*bb_data.shape[0]/(bb_extent[3]-bb_extent[2]))
    bb_y_start = int((s[0] - bb_extent[0])*bb_data.shape[1]/(bb_extent[1]-bb_extent[0]))
    bb_x_w = int(300 *bb_data.shape[0]/(bb_extent[3]-bb_extent[2]))
    bb_y_w = int(300*bb_data.shape[1]/(bb_extent[1]-bb_extent[0]))
    bb_flux = np.sum(bb_data[ bb_x_start:bb_x_start+bb_x_w, bb_y_start:bb_y_start + bb_y_w])
    print(true_flux, clean_flux, bb_flux)
    true_fluxes.append(true_flux)
    clean_fluxes.append(clean_flux/15000)
    bb_fluxes.append(bb_flux)

m2,b2 = np.polyfit(true_fluxes, bb_fluxes, 1)
m1,b1 = np.polyfit(true_fluxes, clean_fluxes, 1)

x = np.arange(min(true_fluxes), max(true_fluxes))
print (x)
axs[1,2].plot(true_fluxes, clean_fluxes, 'ro', label = "WSCLEAN Flux/15000")
axs[1,2].plot(true_fluxes, [m1*x+b1 for x in true_fluxes], 'r-')
axs[1,2].plot(true_fluxes, bb_fluxes, 'bo', label = "Bluebild Flux")
axs[1,2].plot(true_fluxes, [m2*x+b2 for x in true_fluxes], 'b-')
axs[1,2].set(xlabel = "True Flux", ylabel = "Reconstructed Flux")
axs[1,2].legend()

axs[1,2].set_title("Reconstructed Flux")
plt.show()