"""
calculate the labeled array (E0D and L1D) based on the boolean array
autor: Eike E. Köhn
date: Apr 29, 2022
"""
#%% DEFINE THE SCRIPTNAME
scriptdir = '/home/koehne/Documents/publications/paper_future_simulations/scripts/detect_extreme_events/'
scriptname = 'calc_labeled_array.py'

#%% LOAD THE PACKAGES
import xarray as xr
import numpy as np
import scipy.ndimage
from datetime import date
import os

#%% LOAD THE CASES CLASS
exec(open('/home/bblaser/scripts/bsc_beni/eike_reference/modules/define_cases_and_parameters.py').read())

#%% DEFINE FUNCTIONS
def load_boolean_array(params,config,scenario,varia):
    print('Load boolean array.')
    booldir = '/nfs/sea/work/bblaser/data/temp/'
    boolfile = ("boolean_array_hobday2016_threshold_and_climatology_temp_37zlevs_full_1x1meanpool_downsampling_2011-2021analysisperiod_95perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing.nc")
    fn = xr.open_dataset(booldir+boolfile)
    if params.boolean_smoothing_type == 'hobday2016':
        boolean_array = fn.boolean_smoothed_hobday2016.values     
    elif params.boolean_smoothing_type == 'koehn2024':
        boolean_array = fn.boolean_smoothed_koehn2024.values  
    elif params.boolean_smoothing_type == 'no_smoothing':
        boolean_array = fn.boolean_raw.values  
    return boolean_array, fn

def limit_study_area(params,boolean_array,fn):
    print('Limit study region.') # set boolean array in specific regions to 0 (to not include them in detection)
    lat = fn.lat.values
    lon = fn.lon.values
    region_to_set_to_0 = (lon<params.labeled_western_boundary)+(lon>params.labeled_eastern_boundary)+(lat<params.labeled_southern_boundary)+(lat>params.labeled_northern_boundary)
    boolean_array[:,:,region_to_set_to_0] = 0
    return boolean_array

def generate_labeling_structuring_elements():
    s = scipy.ndimage.generate_binary_structure(4,4) # kernel allowing for all connections (across edges and corners)
    # turn of horizontal connections
    sL1D = np.copy(s)
    sL1D[:,:,2,:] = False
    sL1D[:,:,0,:] = False
    sL1D[:,:,:,2] = False
    sL1D[:,:,:,0] = False
    sL1D[:,2,1,1] = True
    sL1D[:,1,1,1] = True
    sL1D[:,0,1,1] = True
    # turn of vertical connections
    sE0D = np.copy(sL1D)
    sE0D[:,0,:,:] = False
    sE0D[:,2,:,:] = False
    return sE0D, sL1D

def apply_ndimage_labeling_function(boolean_array,structuring_element):
    labeled_array,number_of_labels = scipy.ndimage.label(boolean_array,structure=structuring_element)
    if number_of_labels < 2_147_483_647: # maximum number representable in int32
        labeled_array = np.asarray(labeled_array,dtype='int32') # saves storage
    return labeled_array

def generate_labels(boolean_array):
    print('Start labeling the boolean array.')
    sE0D, sL1D = generate_labeling_structuring_elements()
    labeled_E0D = apply_ndimage_labeling_function(boolean_array,sE0D)
    print('E0D labeling done.')
    labeled_L1D = apply_ndimage_labeling_function(boolean_array,sL1D)
    print('L1D labeling done.')
    return labeled_E0D, labeled_L1D 

def save_labeled_arrays(params,labeled_E0D,labeled_L1D,fn,config,scenario,varia):
    # put everything into one dictionary
    print('Set up the dictionary.')
    label_dict = dict()
    label_dict['depth'] = {"dims": ("depth"), "data": fn.depth.values,'attrs': {'units':'m'}}
    label_dict['time'] = {"dims": ("time"), "data": fn.time.values, 'attrs': {'units':'day of year'}}
    label_dict['lat'] = {"dims": ("latitude","longitude"), "data": fn.lat.values, 'attrs': {'units':'degrees N'}}
    label_dict['lon'] = {"dims": ("latitude","longitude"), "data": fn.lon.values, 'attrs': {'units':'degrees E'}}
    label_dict['labeled_L1D'] = {"dims": ("time", "depth", "lat", "lon"), "data": labeled_L1D, 'attrs': {'units':'ID of extreme'}}
    label_dict['labeled_E0D'] = {"dims": ("time", "depth", "lat", "lon"), "data": labeled_E0D, 'attrs': {'units':'ID of extreme'}}

    # generate xarray dataset from dictionary
    print('Generate xarray dataset.')
    ds = xr.Dataset.from_dict(label_dict)

    # add the attributes to xarray dataset
    ds.attrs['author'] = 'E. E. Koehn'
    ds.attrs['date'] = str(date.today())
    ds.attrs['scriptdir'] = scriptdir
    ds.attrs['scriptname'] = scriptname
    ds.attrs['casename'] = casename
    ds.attrs['case_description'] = 'Given by the following attributes:'
    # for pattribute in dir(params):
    #     if not 'keep_only' in pattribute:
    #         if not pattribute.startswith('__') and not callable(getattr(params,pattribute)):
    #             #print(pattribute)
    #             #print(eval('params.'+pattribute))
    #             ds.attrs['params.'+pattribute] = eval('params.'+pattribute)
    #         if not pattribute.startswith('__') and callable(getattr(params,pattribute)):
    #             #print(pattribute)
    #             #print(eval('params.'+pattribute+'()'))
    #             ds.attrs['params.'+pattribute+'()'] = eval('params.'+pattribute+'()')

    # save the xarray dataset to a file
    print("Save to netcdf.")
    if not os.path.exists('{}{}'.format(params.dir_root_labeled_arrays,config)):
        os.mkdir('{}{}'.format(params.dir_root_labeled_arrays,config))
        print('{}{} created. '.format(params.dir_root_labeled_arrays,config))
    else:
        print('{}{} exists already. Do nothing. '.format(params.dir_root_labeled_arrays,config))
    if not os.path.exists('{}{}/{}'.format(params.dir_root_labeled_arrays,config,scenario)):
        os.mkdir('{}{}/{}'.format(params.dir_root_labeled_arrays,config,scenario))
        print('{}{}/{} created.'.format(params.dir_root_labeled_arrays,config,scenario))
    else:
        print('{}{}/{} exists already. Do nothing.'.format(params.dir_root_labeled_arrays,config,scenario))
    savepath = '/nfs/sea/work/bblaser/data/temp/' #'{}{}/{}/'.format(params.dir_root_labeled_arrays,config,scenario)

    #savepath = params.dir_labeled_arrays 
    netcdfname = params._fname_labeled_array(varia)
    ds.to_netcdf(savepath+netcdfname)
    print("Saving done")

def calc_labeled_array(params,config,scenario,varia):
    boolean_array, fn  = load_boolean_array(params,config,scenario,varia)
    boolean_array = limit_study_area(params,boolean_array,fn)
    labeled_E0D, labeled_L1D = generate_labels(boolean_array)
    return labeled_E0D, labeled_L1D, fn



#%% DEFINE CASE AND START THE CALCULATION
casedir = '/home/bblaser/scripts/bsc_beni/eike_reference/modules/cases/'
#casedir = '/home/koehne/Documents/publications/paper_future_simulations/scripts/modules/cases/'
casenames = ['case00.yaml']#['case00_reference.yaml']

config = 'romsoc_fully_coupled' #'romsoc_fully_coupled'
scenario = 'present' #'ssp585'
varia = 'temp'

for casename in casenames:
    params = read_config_files(casedir+casename)
    #labeled_E0D, labeled_L1D, fn = calc_labeled_array(params,config,scenario,varia)
    boolean_array, fn  = load_boolean_array(params,config,scenario,varia)
    boolean_array = limit_study_area(params,boolean_array,fn)
    labeled_E0D, labeled_L1D = generate_labels(boolean_array)
    #%%
    save_labeled_arrays(params,labeled_E0D,labeled_L1D,fn,config,scenario,varia)

# %%
