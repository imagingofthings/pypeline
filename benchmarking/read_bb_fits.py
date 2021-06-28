from astropy.io import fits
import astropy.wcs as pywcs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import numpy as np
from scipy import ndimage
import sys

'''
def pixels_to_sky(self, i, j, k):
    coords = self.w.wcs_pix2world([[ i, j, k]], 0)[0]
    return coords

def sky_to_pixels(self, ra, dec, freq):
    coords = self.w.wcs_world2pix([[ ra, dec, freq]], 0)[0]
    return coords
'''

def convert_coords(coords, wcs1, wcs2):
    print(coords)
    coords = [ [*c, 0,0] for c in coords]
    cel_coords = wcs1.wcs_pix2world(coords, 0)
    print(cel_coords)
    coords = wcs2.wcs_world2pix(cel_coords, 0)
    print(coords)
    coords = [ [ int(c[0]), int(c[1])] for c in coords]
    return coords

with fits.open("/home/etolley/bluebild/pypeline/bluebild_ss_4gauss_37Stations.fits") as hdul:
    bb_data = hdul[1].data
    bb_wcs = pywcs.WCS(hdul[1].header)


with fits.open("/home/etolley/casacore_setup/deconvScale-image.fits") as hdul: #
    clean_data = hdul[0].data[0,0,:,:]
    clean_wcs = pywcs.WCS(hdul[0].header)

with fits.open("/home/etolley//data/gauss4/gauss4-image-pb.fits") as hdul: #"/home/etolley/casacore_setup/deconv-image.fits"
    #clean_data = hdul[0].data[0,0,:,:]
    true_wcs = pywcs.WCS(hdul[0].header)

with fits.open("/home/etolley/data/gauss4/C_4gaussian-model.fits") as hdul:
    true_data = hdul[0].data
    #true_wcs = pywcs.WCS(hdul[0].header)

avg_true_data_val = np.mean(true_data)
regions = ndimage.find_objects(ndimage.label(true_data > avg_true_data_val)[0])
print(regions)

'''
bb_mask = np.zeros(bb_data.shape)
for i in range(4):
    bb_mask[i,:,:] /= np.max(bb_mask[i,:,:] )

bb_data = bb_mask
'''


import copy
my_cmap = copy.copy(cm.get_cmap("GnBu_r")) # copy the default cmap
my_cmap.set_bad((0,0,0))

bb_data_sum = np.sum(bb_data,axis = 0)

fig, axs = plt.subplots(2, 4)
axs = axs.flatten()
plt.subplots_adjust(bottom=0.05, top=0.95, wspace=0.35, hspace = 0.2)

axs[0].imshow(bb_data_sum,  cmap = my_cmap, extent = [0, 2000, 0, 2000], origin='lower')
axs[0].set_title("Bluebild Combined")
axs[1].imshow(clean_data, cmap = my_cmap, origin='lower')
axs[1].set_title("WSCLEAN")
axs[2].imshow(true_data, cmap = my_cmap, origin='lower')
axs[2].set_title("Truth")

for i in range(4):
    axs[4+i].imshow(bb_data[i],  cmap = my_cmap, extent = [0, 2000, 0, 2000], origin='lower')
    axs[4+i].set_title("Bluebild Level {0}".format(i))
avg_level_fluxes = [np.mean(bb_data[i]) for i in range(4)]

true_fluxes, clean_fluxes, bb_fluxes, bb_level_fluxes = [],[],[],[]
margin = 50

for r in regions:
    # true coordinates
    x1, x2, y1, y2 = r[0].start-margin, r[0].stop+margin, r[1].start-margin, r[1].stop+margin
    clean1, clean2 = convert_coords([ [x1,y1],[x2,y2]], true_wcs, clean_wcs)
    #print(x1,x2,y1,y2)
    axs[0].add_patch(plt.Rectangle((y1,x1), y2-y1, x2-x1, fill = False, edgecolor="r"))
    axs[1].add_patch(plt.Rectangle((clean1[1],clean1[0]), clean2[1]-clean1[1], clean2[0]-clean1[0], fill = False, edgecolor="r"))
    axs[2].add_patch(plt.Rectangle((y1,x1), y2-y1, x2-x1, fill = False, edgecolor="r"))

    bb_level_flux = 0
    for i in range(4):
        sample_flux = np.mean(bb_data[i, int(x1/10) : int(x2/10), int(y1/10): int(y2/10)])
        print(i, sample_flux > avg_level_fluxes[i] )
        if sample_flux > avg_level_fluxes[i]:
            #axs[i+4].add_patch(plt.Rectangle((y1,x1), y2-y1, x2-x1, fill = False, edgecolor="r"))
            bb_level_flux += sample_flux

    true_fluxes.append(  np.sum(true_data[x1:x2, y1:y2]))
    clean_fluxes.append( np.sum(clean_data[clean1[0]:clean2[0], clean1[1]:clean2[1]]))
    bb_fluxes.append(    np.sum(bb_data_sum[ int(x1/10) : int(x2/10), int(y1/10): int(y2/10)]))
    bb_level_fluxes.append(bb_level_flux)

print(bb_level_fluxes)
clean_fluxes = np.array(clean_fluxes) / max(clean_fluxes) 
bb_level_fluxes = np.array(bb_level_fluxes) / max(bb_level_fluxes) 
bb_fluxes = np.array(bb_fluxes) / max(bb_fluxes) 
x = np.arange(min(true_fluxes), max(true_fluxes))

m2,b2 = np.polyfit(true_fluxes, bb_fluxes, 1)
m1,b1 = np.polyfit(true_fluxes, clean_fluxes, 1)

axs[3].plot(true_fluxes, clean_fluxes, 'ro', label = "WSCLEAN Flux")
axs[3].plot(true_fluxes, [m1*x+b1 for x in true_fluxes], 'r-')
axs[3].plot(true_fluxes, bb_fluxes, 'bo', label = "Combined Bluebild Flux")
axs[3].plot(true_fluxes, [m2*x+b2 for x in true_fluxes], 'b-')
axs[3].set(xlabel = "True Flux", ylabel = "Normalized Reconstructed Flux")
axs[3].legend()

axs[3].set_title("Reconstructed Flux")

plt.show()

###########################################################################

sys.exit()
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