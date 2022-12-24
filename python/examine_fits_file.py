from astropy.io import fits
import astropy.wcs as pywcs
import matplotlib.pyplot as plt
import numpy as np

def pixels_to_sky(w, i, j):
        coords = w.wcs_pix2world([[ i, j, 0]], 0)[0]
        return coords[0], coords[1]

def sky_to_pixels(w, ra, dec):
    coords = w.wcs_world2pix([[ ra, dec, 0]], 0)[0]
    return coords[0], coords[1]

with fits.open("bluebild_ss_image1.fits") as hdul:
    hdul.info()

    # the cards of the file
    print(repr(hdul[0].header))
    print(repr(hdul[1].header))

    w = pywcs.WCS(hdul[1].header)
    data = hdul[1].data
    print(data.shape)
    img_extent = [0, data.shape[2], 0, data.shape[1]]

    print("Lower corner:", pixels_to_sky(w,0,0))
    print("Upper corner:", pixels_to_sky(w,data.shape[2],data.shape[1]))

    plt.figure()
    plt.imshow( np.abs(data[0,:,:]), origin='lower', extent = img_extent, cmap = 'bone')
    plt.colorbar()
    plt.xlabel("Y")
    plt.ylabel("X")
    #plt.savefig("mygraph.png")
    plt.show()