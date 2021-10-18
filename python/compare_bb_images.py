'''
Run this script after generating .npy files with test_bluebild.py
'''

import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import pypeline.phased_array.data_gen.source as source
import imot_tools.math.sphere.transform as transform
import matplotlib.pyplot as plt
import imot_tools.io.s2image as s2image

def get_catalog_locations(img, catalog, projection = "AEQD"):
    print("Now calculating source catalog positions...")
    proj = img._draw_projection(projection)
    N_src = catalog.size // 3
    if not (catalog.shape == (3, N_src)):
        raise ValueError("Parameter[catalog]: expected (3, N_src) array.")

    _, grid_x, grid_y = img._grid
    _, cat_x, cat_y = catalog

    cat_i = []
    cat_j = []
    for x,y in zip(cat_x, cat_y):
        
        min_i = min_j = -1 
        minval = 1000
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                dx = np.abs(grid_x[i][j] - x)
                dy = np.abs(grid_y[i][j] - y)
                #print (" ",i, j, dx, dy)
                if dx + dy < minval:
                    minval = dx+dy
                    min_i = i
                    min_j = j
        print(x,y, '=>', min_i, min_j)
        cat_i.append(min_i)
        cat_j.append(min_j)

    return cat_i, cat_j

field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
FoV, frequency = np.deg2rad(8), 145e6
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=30)



data_nu = np.load("bluebild_nufft_img.npy")
data_ss = np.load("bluebild_ss_img.npy")        
data_ps = np.load("bluebild_ps_img.npy")
grid = np.load("bluebild_np_grid.npy")


data_nu = np.sum(data_nu, axis=0)
data_ss = np.sum(data_ss, axis=0)
data_ps = np.sum(data_ps, axis=0)
data_nu /= np.max(data_nu)
data_ss /= np.max(data_ss)
data_ps /= np.max(data_ps)

img_nu = s2image.Image(data_nu, grid)

source_x, source_y = get_catalog_locations(img_nu, sky_model.xyz.T)

fig, ax = plt.subplots(ncols=3, nrows = 2, figsize=(16, 8))
ax = ax.flatten()

def draw(data, a, title):
    pos = a.imshow(data, origin = 'lower')
    fig.colorbar(pos, ax=a)
    a.set_title(title)

draw(data_nu, ax[0], 'NUFFT')
draw(data_ss, ax[1], 'Standard Synthesis')
draw(data_ps, ax[2], 'Periodic Synthesis')
draw(data_ss/np.max(data_ss)-data_nu/np.max(data_nu), ax[3], 'SS-NUFFT')
draw(data_ss/np.max(data_ss)-data_ps/np.max(data_ps), ax[4], 'SS-PS')

ax[0].scatter(source_y, source_x, s=40, facecolors='none', edgecolors='r')
ax[1].scatter(source_y, source_x, s=40, facecolors='none', edgecolors='r')
ax[2].scatter(source_y, source_x, s=40, facecolors='none', edgecolors='r')
for x,y, intensity in zip(source_x, source_y, sky_model.intensity):
    print(intensity, np.sum(data_nu[x-4:x+4,y-4:y+4]))

measurement1_nu =  [np.max(data_nu[x-4:x+4,y-4:y+4]) for x,y in zip(source_x, source_y)]
measurement1_ps =  [np.max(data_ps[x-4:x+4,y-4:y+4]) for x,y in zip(source_x, source_y)]
measurement1_ss =  [np.max(data_ss[x-4:x+4,y-4:y+4]) for x,y in zip(source_x, source_y)]

ax[5].scatter(sky_model.intensity, measurement1_nu, c = 'r', label = 'nufft')
ax[5].scatter(sky_model.intensity, measurement1_ps, c = 'g', label = 'ps')
ax[5].scatter(sky_model.intensity, measurement1_ss, c = 'b', label = 'ss')
ax[5].set_xlabel('source intensity')
ax[5].set_ylabel('measured flux at source position')
ax[5].legend()
plt.savefig("compare_bb_images")

