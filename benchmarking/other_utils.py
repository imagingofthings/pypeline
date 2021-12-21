import numpy as np
import typing as typ
import pypeline.util.frame as frame

def rad_average(data, cen_x=None, cen_y=None, bin_size=1):
    # Image center
    if(cen_x==None and cen_y==None):
        cen_x = data.shape[1]//2
        cen_y = data.shape[0]//2
    else:
        pass

    # Find radial distances
    [X, Y] = np.meshgrid(np.arange(data.shape[1]) - cen_x, np.arange(data.shape[0]) - cen_y)
    #R = np.sqrt(np.square(X) + np.square(Y))
    R = X
    rad = np.arange(1, np.max(R), 1)
    intensity = np.zeros(len(rad))

    index= 0
    for i in rad:
        mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
        values = data[mask]
        intensity[index] = np.mean(values)
        index += 1
    return intensity, rad

def RescaleData(arr, a=-1, b=1):
    scaled_arr = (arr.astype(np.float32) - np.min(arr))/(np.max(arr) - np.min(arr)) * (b-a) + a
    return scaled_arr


def nufft_make_grids(FoV, grid_size, field_center) -> typ.Tuple[np.ndarray, np.ndarray]:
    r"""
    Imaging grid.

    Returns
    -------
    lmn_grid, xyz_grid: Tuple[np.ndarray, np.ndarray]
        (3, grid_size, grid_size) grid coordinates in the local UVW frame and ICRS respectively.
    """
    lim = np.sin(FoV / 2)
    grid_slice = np.linspace(-lim, lim, grid_size)
    l_grid, m_grid = np.meshgrid(grid_slice, grid_slice)
    n_grid = np.sqrt(1 - l_grid ** 2 - m_grid ** 2)  # No -1 if r on the sphere !
    lmn_grid = np.stack((l_grid, m_grid, n_grid), axis=0)
    uvw_frame = frame.uvw_basis(field_center)
    xyz_grid = np.tensordot(uvw_frame, lmn_grid, axes=1)
    return lmn_grid, xyz_grid