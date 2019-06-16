# #############################################################################
# plot.py
# =======
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
`Matplotlib <https://matplotlib.org/>`_ helpers.
"""

import pathlib

import matplotlib.colors as col
import pandas as pd
import pkg_resources as pkg

import pypeline.util.argcheck as chk


@chk.check(dict(name=chk.is_instance(str), N=chk.allow_None(chk.is_integer)))
def cmap(name, N=None):
    """
    Load one of Pypeline's custom colormaps.

    All maps are defined under ``<pypeline_dir>/data/colormap/``.

    Parameters
    ----------
    name : str
        colormap name.
    N : int, optional
        Number of color levels. (Default: all).

        If `N` is smaller than the number of levels available in the colormap, then the last `N`
        colors will be used.

    Returns
    -------
    colormap : :py:class:`~matplotlib.colors.ListedColormap`

    Examples
    --------
    .. doctest::

       import numpy as np
       import matplotlib.pyplot as plt

       from pypeline.util.plot import cmap

       x, y = np.ogrid[-1:1:100j, -1:1:100j]

       fig, ax = plt.subplots(ncols=2)
       ax[0].imshow(x + y, cmap='jet')
       ax[0].set_title('Jet')

       ax[1].imshow(x + y, cmap=cmap('matthieu-custom-sky'))
       ax[1].set_title('Matthieu-Custom-Sky')

       fig.show()

    .. image:: _img/cmap_example.png
    """
    if (N is not None) and (N <= 0):
        raise ValueError("Parameter[N] must be a positive integer.")

    cmap_rel_dir = pathlib.Path("data", "colormap")
    cmap_rel_path = cmap_rel_dir / f"{name}.csv"

    if pkg.resource_exists("pypeline", str(cmap_rel_path)):
        cmap_abs_path = pkg.resource_filename("pypeline", str(cmap_rel_path))
        colors = pd.read_csv(cmap_abs_path).loc[:, ["R", "G", "B"]].values

        N = len(colors) if (N is None) else N
        colormap = col.ListedColormap(colors[-N:])
        return colormap

    else:  # no cmap under that name.
        # List available cmaps.
        cmap_names = [
            pathlib.Path(_).stem
            for _ in pkg.resource_listdir("pypeline", str(cmap_rel_dir))
            if _.endswith("csv")
        ]
        raise ValueError(
            f"{name} is not a pypeline-defined colormap. " f"Available options: {cmap_names}"
        )
