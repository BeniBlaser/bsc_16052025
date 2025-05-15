"""
this script serves to load ETOPO data
in additional step, the ETOPO data is shuffeled, to have it centered on the Pacific 
author: Eike E. Koehn
date: May 13, 2020
"""

import numpy as np
import netCDF4

# load the ETOPO data
topofile='ETOPO15_mean_bath.nc'
topopath='/nfs/kryo/work/updata/bathymetry/ETOPO1/'
topodata = netCDF4.Dataset(topopath+topofile)
topo = topodata.variables['mean_bath'][:]
topo_lon = topodata.variables['longitude'][:]+360
topo_lat = topodata.variables['latitude'][:]

# shuffle the data to have it centered on the Pacific
topo_lon[topo_lon>360]=topo_lon[topo_lon>360]-360
topo_lon_shuffle = np.hstack((topo_lon[int(len(topo_lon)/2):],topo_lon[:int(len(topo_lon)/2)]))
topo_shuffle = np.hstack((topo[:,int(len(topo_lon)/2):],topo[:,:int(len(topo_lon)/2)]))

topo_shuffle[topo_shuffle>0]=1
topo_shuffle[topo_shuffle<0]=np.NaN
