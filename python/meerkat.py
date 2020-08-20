from tqdm import tqdm as ProgressBar
import astropy.units as u
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants

import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_fd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.measurement_set as measurement_set

#/scratch/foureste/Meerkat
#1524929477.ms  1524947605.ms  1532022061.ms  1532552470_sdp_l0_1284.full_pol.ms

# Instrument
N_station = 24
ms_file = "/scratch/foureste/Meerkat/1524929477.ms"
ms = measurement_set.LofarMeasurementSet(ms_file, N_station)
gram = bb_gr.GramBlock()