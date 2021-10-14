import numpy as np

def rad_average(data, bin_size=1):
    # Image center
    cen_x = data.shape[1]//2
    cen_y = data.shape[0]//2

    # Find radial distances
    [X, Y] = np.meshgrid(np.arange(data.shape[1]) - cen_x, np.arange(data.shape[0]) - cen_y)
    R = np.sqrt(np.square(X) + np.square(Y))
    rad = np.arange(1, np.max(R), 1)
    intensity = np.zeros(len(rad))

    index= 0
    for i in rad:
        mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
        values = data[mask]
        intensity[index] = np.mean(values)
        index += 1
    return intensity, rad