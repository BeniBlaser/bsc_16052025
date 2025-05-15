"""
This file serves to get model grid information in the standard fashion, which will be used for the analyses.
should be imported into another script by executing:
exec(open('get_model_grid.py').read())

author: Eike E. Koehn
date: May 13, 2020
"""

#### Load necessary modules
###########################
import xarray as xr
# import modules written by Martin Frischknecht (ETH Zürich) for processing ROMS model data
#exec(open('/home/koehne/Documents/scripts/python/martinfr/myromstools.py').read())


#### Define the get_model_grid function
#######################################
def get_model_grid(gridpath,gridfile,outpath,outfile_example):
    #%% load the grid
    # Read grid
#    gridpath = '/net/kryo/work/koehne/roms/inputs/humpac15/hindcast_1979_2016/grd/'
#    gridfile = 'humpac15_grd.nc'
#    filepath = '/net/kryo/work/koehne/roms/output/humpac15/hindcast_1979_2016/hindcast_r016_humpac15/daily/avg/'
#    example_filename = 'humpac15_1979_avg.nc'
    
    # get the grid
    romsGrd = getGrid(gridpath+gridfile)
    # Add necessary attributes
    romsGrd.getAttrs(outpath+outfile_example)
    # Add lat lon
    romsGrd.getLatLon()
    # Add grid area
    romsGrd.getArea()
    # Add angle
    romsGrd.getAngle()

    # open the grid file
    grid = xr.open_dataset(gridpath+gridfile)

    return romsGrd,grid


def get_model_z_dz(fpin,fpin_grd,NZ):
    dz = compute_dz(fpin,fpin_grd,NZ,zeta=None,stype=3)
    z = compute_zlev(fpin,fpin_grd,NZ,'r',zeta=None,stype=3)
    zw = compute_zlev(fpin,fpin_grd,NZ,'w',zeta=None,stype=3)
    return z,dz,zw



#### the __name__ == '__main__' case
################################
if __name__ == '__main__':
    romsGrd,grid = get_model_grid(gridpath,gridfile,outpath,outfile_example)
    fpin = xr.open_dataset(outpath+outfile_example)
    z,dz,zw = get_model_z_dz(fpin,grid,romsGrd.NZ)
