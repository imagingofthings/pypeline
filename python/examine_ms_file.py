import casacore.tables as ct
import astropy.units as u
import pathlib

def examine(filename):

	print("ms file:", filename)

	path = pathlib.Path(filename).absolute()

	if not path.exists():
	    raise FileNotFoundError(f"{filename} does not exist.")

	if not path.is_dir():
	    raise NotADirectoryError(f"{filename} is not a directory, so cannot be an MS file.")

	print ("Columns are:")
	print(ct.taql(f"select * from {filename}").colnames())

	print ("Frequencies are:")
	query = f"select CHAN_FREQ, CHAN_WIDTH from {filename}::SPECTRAL_WINDOW"
	table = ct.taql(query)
	freq = table.getcell("CHAN_FREQ", 0).flatten() * u.Hz
	print(freq)


ms_files = ["/home/etolley/data/gauss_emma/gauss1hr_t201806301100_SBL150.MS"]
for m in ms_files: examine(m)



