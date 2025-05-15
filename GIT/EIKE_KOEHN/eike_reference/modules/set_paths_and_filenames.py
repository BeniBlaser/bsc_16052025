"""
This file serves to set the paths and filenames pointing to general model files, which will be used for the analyses.
can be called from another script by executing:
exec(open('set_paths_and_filenames.py').read())

author: Eike E. Koehn
date: May 13, 2020
"""

host = '/nfs/kryo/'

# Grid files
gridpath = host+'work/loher/ROMSOC/grd/'
gridfile = 'pactcs30_grd.nc'

# Model output files
outpath = host+'work/loher/ROMS/future_sim/hind_isoneutral_swf_coupling_region/avg/'
outfile_example = 'pactcs30_2010_avg.nc'

