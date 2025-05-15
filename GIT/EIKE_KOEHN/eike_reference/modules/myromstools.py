#!/usr/bin/env python
# author: Martin Frischknecht
# Core functions used to extract grid information or to save netcdf files 

import netCDF4
import numpy as np

#########################################################
# Class for ROMS grd and clm files
# (For use in various post-processing scripts)
#########################################################

class getGrid(object):
    '''
    Read the basics of ROMS setup into class for further use in other functions
    and classes.
    '''
    # Read grid file
    def __init__(self,grdfile):
        # Set grd file
        self.grdfile = grdfile
        self.ncgrd = netCDF4.Dataset(grdfile, mode='r')
        # Read mask
        self.mask_rho = self.ncgrd.variables['mask_rho'][:]
        self.FillValue = getattr(self.ncgrd.variables['mask_rho'],'_FillValue')
	# Read dimensions
        self.SY = self.mask_rho.shape[0]
        self.SX = self.mask_rho.shape[1]
        
    def getAttrs(self,clmfile):
        # Set clm file
        self.ncclm  = netCDF4.Dataset(clmfile, mode='r')
        # Read attributes
        try:
            self.theta_s = getattr(self.ncclm,'theta_s')
            self.theta_b = getattr(self.ncclm,'theta_b')
            self.hc      = getattr(self.ncclm,'hc')
        except AttributeError:
            self.theta_s = self.ncclm.variables['theta_s'][0]
            self.theta_b = self.ncclm.variables['theta_b'][0]
            self.hc      = self.ncclm.variables['hc'][0]            
        # Vertical dimension
        self.NZ       = self.ncclm.dimensions['s_rho'].size
    
    def setClmFiles(self,clmfile,clm2file):
	# Set clm file
        if not hasattr(self, 'ncclm'):
            self.ncclm  = netCDF4.Dataset(clmfile, mode='r')
        # Set clm2 file
        self.ncclm2 = netCDF4.Dataset(clm2file, mode='r')

    def getTopo(self):
        # Read topography
        self.h     = self.ncgrd.variables['h'][:]
        self.hmin  = getattr(self.ncgrd,'hmin')
        self.hmax  = getattr(self.ncgrd,'hmax')
        
    def getLatLon(self):
        # Read Lat/Lon
        self.lon_rho  = self.ncgrd.variables['lon_rho'][:]
        self.lat_rho  = self.ncgrd.variables['lat_rho'][:]
        
    def getArea(self):
        # Read pm/pn
        self.area  = 1/(self.ncgrd.variables['pm'][:]*self.ncgrd.variables['pn'][:])

    def getAngle(self):
        # Read angle
        self.angle  = self.ncgrd.variables['angle'][:]

#########################################################
# Vertical sigma level depths and spacing
#########################################################

def compute_zlev(fpin,fpin_grd,NZ,type,zeta=None,stype=3):
    # Compute z levels of rho points for ZERO SSH. Input:
    #
    #  fpin: file descriptor pointing to a NetCDF file containing theta_b,
    #        theta_s and Tcline or hc
    #  fpin_grd: file descriptor pointing to a NetCDF file containing h
    #  NZ: number of vertical (rho) levels
    #  type:  'r': rho points
    #         'w': w points
    #  stype: specifies type of sigma levels used:
    #          1: similar to Song, Haidvogel 1994
    #          2: Shchepetkin 2006
    #          3: Shchepetkin 2010 (or so)

    import numpy as np
    import sys
    
    h = fpin_grd.variables['h'][:,:]
    try:
        theta_b = fpin.theta_b
        theta_s = fpin.theta_s
    except AttributeError:
        # theta_b/s may be variables:
        theta_b = fpin.variables['theta_b'][0]
        theta_s = fpin.variables['theta_s'][0]
        
    if stype == 1:
        hmin = min(min(h))
        try:
            Tcline = fpin.Tcline
            hc = min(hmin,Tcline)
        except AttributeError:
            hc = fpin.hc
            hc = min(hmin,hc)
    elif stype == 2 or stype == 3:
        try:
            hc = fpin.hc
        except AttributeError:
            # hc may be a variable:
            hc = fpin.variables['hc'][0]
    else:
        msg = '%s: Unknown type of sigma levels'.format(stype)
        sys.exit(msg)
    ds = 1./NZ  # float, to prevent integer division in sc
    if type == 'w':
        lev = np.arange(NZ+1)
        sc = (lev - NZ) * ds
        nr_zlev = NZ+1 # number of vertical levels
    else:
        lev = np.arange(1,NZ+1)
        sc = -1 + (lev-0.5)*ds
        nr_zlev = NZ # number of vertical levels
    Ptheta = np.sinh(theta_s*sc)/np.sinh(theta_s)
    Rtheta = np.tanh(theta_s*(sc+.5))/(2*np.tanh(.5*theta_s))-.5
    if stype <= 2:
        Cs = (1-theta_b)*Ptheta+theta_b*Rtheta
    elif stype == 3:
        if theta_s > 0:
            csrf=(1.-np.cosh(theta_s*sc))/(np.cosh(theta_s)-1.)
        else:
            csrf=-sc**2
        if theta_b > 0:
            Cs=(np.exp(theta_b*csrf)-1.)/(1.-np.exp(-theta_b))
        else:
            Cs=csrf
    z0 = np.zeros((nr_zlev,h.shape[0],h.shape[1]))#,np.float)
    if stype == 1:
        cff = (sc-Cs)*hc
        cff1 = Cs
        hinv = 1.0 / h
        for k in range(nr_zlev):
            z0[k,:,:] = cff[k]+cff1[k]*h
            if not (zeta is None):
                z0[k,:,:] = z0[k,:,:]+zeta*(1.+z0[k,:,:]*hinv)
    elif stype == 2 or stype == 3:
        hinv = 1.0/(h+hc)
        cff = hc*sc
        cff1 = Cs
        for k in range(nr_zlev):
            tmp1 = cff[k]+cff1[k]*h
            tmp2 = np.multiply(tmp1,hinv)
            if zeta is None:
                z0[k,:,:] = np.multiply(h,tmp2)
            else:
                z0[k,:,:] = zeta + np.multiply((zeta+h),tmp2)
    # Return
    return z0

def compute_dz(fpin,fpin_grd,NZ,zeta=None,stype=3):
  
    # Compute dz of sigma level rho points for ZERO SSH. Input:
    #
    #  fpin: file descriptor pointing to a NetCDF file containing theta_b,
    #        theta_s and Tcline or hc
    #  fpin_grd: file descriptor pointing to a NetCDF file containing h
    #  NZ: number of vertical (rho) levels
    #  stype: specifies type of sigma levels used:
    #          1: similar to Song, Haidvogel 1994
    #          2: Shchepetkin 2006
    #          3: Shchepetkin 2010 (or so)
    
    # Compute depth of w sigma levels
    depth_w = -compute_zlev(fpin,fpin_grd,NZ,type='w',zeta=zeta,stype=3)
    
    # Compute dz between w sigma levels (= dz of sigma layer)
    dz_sigma = depth_w[:-1]-depth_w[1:]
    
    return dz_sigma



#########################################################
# Writing processed data to NetCDF file of ROMS dimensions
#########################################################

def writeNetCDF(romsGrd,dData,outfile,dDims=None,dAttrs=None,LatLon=False):
    '''
    Writes given data with ROMS grid dimensions to NetCDF file.
    
    Inputs
    romsGrd: grid class with necessary attributes
    dData:   data dictionary containing data for different variables
             (e.g. dData['temp'],dData['salt'])
    dDims:   dictionary containing the dim/len(dim) of the variables
    outfile: specified path and filename for output as string
             (e.g. '/net/kryo/work/martinfr/Data/pacsg/slavg/new_file_name.nc')
    
    Outputs
    Data is stored to NetCDF file at given directory and filename.
    
    Martin Frischknecht, 2014
    '''
    
    import datetime
    
    # Create dimensions
    if dDims==None:
        print('Please provide dimensions of data...')
        return
    else:
        # Open new NetCDF file to write to
        ncfile = netCDF4.Dataset(outfile, 'w', format='NETCDF4')
        
        # Take a variable as reference for dimensions
        for key in dDims.keys():
            for dim,dimlen in zip(dDims[key]['dims'],dDims[key]['dimlens']):
                if not dim in ncfile.dimensions:
                    ncfile.createDimension(dim,dimlen)
            
    # Set author attributes
    setattr(ncfile, 'author', 'koehne')
    setattr(ncfile, 'created', str(datetime.date.today()))
    
    # Writing variable data
    for var in dData.keys():
        print('Writing {}...'.format(var))
        sz = dData[var].shape
        try:
            data = ncfile.createVariable(var,'f4',dDims[var]['dims'],fill_value=dData[var].fill_value)
        except AttributeError:
            data = ncfile.createVariable(var,'f4',dDims[var]['dims'])
        # If 4D variable do sequential writing
        if len(sz)>3:
            for t in range(sz[0]):
                data[t] = dData[var][t]
        else:
            data[:] = dData[var][:]
        # Set atrributes
        if not dAttrs==None:            
            for attr in dAttrs[var].keys():
                if attr=='_FillValue':
                    pass
                else:
                    setattr(data, attr, dAttrs[var][attr])
		    
    # Write lon-lat data
    if LatLon==True:
        for var in ['lon_rho','lat_rho']:
            varDims = ('eta_rho','xi_rho')
            data = ncfile.createVariable(var,'f4',varDims,fill_value=romsGrd.ncgrd.variables[var]._FillValue)
            data[:] = romsGrd.ncgrd.variables[var][:]
            for attr in romsGrd.ncgrd.variables[var].ncattrs():
                if attr=='_FillValue':
                    pass
                else:
                    setattr(data, attr, getattr(romsGrd.ncgrd.variables[var],attr)) 
    # Close the file and end the program.
    ncfile.close()
