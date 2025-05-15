"""
calculate the boolean array based on the threshold file
autor: Eike E. Köhn
date: Apr 29, 2022

running on Euler on new software stack (type "env2lmod" to activate), with gcc/8.2.0, python/3.11.2 -> 
I installed the dataclasses package using "pip install --user dataclasses"

sbatch --time=24:00:00 --mem-per-cpu=750G -n 1 --wrap="python calc_boolean_array.py"

"""

#%% DEFINE THE SCRIPTNAME
scriptdir = '/cluster/home/koehne/uphome/Documents/publications/paper_future_simulations/scripts/detect_extreme_events/'
scriptname = 'calc_boolean_array.py'

#%% LOAD THE PACKAGES
import xarray as xr
import numpy as np
import scipy.ndimage
from datetime import date
import os

#%% LOAD THE CASES CLASS
exec(open('../modules/define_cases_and_parameters.py').read())

#%% DEFINE FUNCTIONS
def load_threshold_and_climatology(params,config,scenario,varia,percentile):
    threshdir = '/nfs/sea/work/bblaser/data/temp/' #'{}{}/{}/'.format(params.dir_root_thresholds_and_climatologies,config,scenario) #params.dir_thresholds_and_climatologies      #'/nfs/kryo/work/koehne/roms/analysis/humpac15/hindcast_1979_2019/hindcast_r105_humpac15/thresholds_and_climatologies/perc{}/'.format(perclevel)
    threshfile = 'hobday2016_threshold_and_climatology_temp_37zlevs_full_1x1meanpool_downsampling_2011-2021analysisperiod_80perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing.nc' #'hobday2016_threshold_and_climatology_temp_37zlevs_full_1x1meanpool_downsampling_2011-2021analysisperiod_95perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing.nc'  #params._fname_threshold_and_climatology(varia).replace(f'{params.threshold_percentile}perc',f'{percentile}perc') #params._fname_threshold_and_climatology()   #'hobday2016_threshold_and_climatology_fac{}_perc{}_baseperiod_{}-{}.nc'.format(fac,perclevel,base_period_start,base_period_end)
    fn = xr.open_dataset(threshdir+threshfile)
    print('Load the fields:')
    thresh_365 = fn.thresh_smoothed.values
    clim_365 = fn.clim_smoothed.values
    normalizer_365 = fn.intensity_normalizer_smoothed.values
    return thresh_365, clim_365, normalizer_365

def include_feb29(data_365):
    data_feb29 = np.mean(data_365[59:61,:,:,:],axis=0,keepdims=True)
    data_366 = np.concatenate((data_365[:60,:,:,:],data_feb29,data_365[60:,:,:,:]),axis=0)
    return data_366

def calculate_366_arrays(thresh_365, clim_365, normalizer_365):
    thresh_366 = include_feb29(thresh_365)
    clim_366 = include_feb29(clim_365)
    normalizer_366 = include_feb29(normalizer_365)
    return thresh_366, clim_366, normalizer_366

def retrieve_model_file(parameters,year,config,scenario):
    model_file_dir = eval('parameters.dir_{}_{}_daily_zlevs'.format(config,scenario))
    if config == 'roms_only':
        model_file = parameters._fname_model_output_regridded().replace('YYYY',str(year))
        return_val = model_file_dir+model_file
    elif config == 'romsoc_fully_coupled':
        dummy_file_fname = parameters._fname_model_output_regridded_romsoc().replace('YYYY',str(year))
        return_val = [model_file_dir+dummy_file_fname.replace('MMM','{:03d}'.format(int(mon+1))) for mon in range(12)]
    return return_val

def open_yearfile(model_file,config):
    if config == 'roms_only':
        filehandle = xr.open_dataset(model_file)
    elif config == 'romsoc_fully_coupled':
        filehandle = xr.open_mfdataset(model_file,concat_dim='time',combine='nested',parallel=True,data_vars='minimal')
    return filehandle

def generate_and_concatenate_annual_boolean_files(params, thresh_365, clim_365, normalizer_365, thresh_366, clim_366, normalizer_366, config, scenario, varia,percentile):
    # loop over regridded model output and compare with threshold to generate boolean. put each years boolean into a list
    all_boolean_list = []
    all_intensity_list = []
    all_cat_list = []
    for yidx,year in enumerate(range(params.threshold_period_start_year,params.threshold_period_end_year+1)):
        print(year)
        model_file = retrieve_model_file(params,year,config,scenario)
        mfn = open_yearfile(model_file,config)
        print('load the variable for the year')
        if varia == 'Hions':
            mvar = 10**(-1*mfn.variables['pH_offl'].values)
        else:
            mvar = mfn.variables[varia].values
        print('calculate the boolean')
        # Treat the regular year case
        if np.shape(mvar)[0]==365:
            if percentile > 50:
                boolean = mvar > thresh_365
            elif percentile < 50:
                boolean = mvar < thresh_365
            intensity = np.asarray(mvar - thresh_365,dtype='float16')
            cat_index = np.asarray((mvar - clim_365)/normalizer_365,dtype='float16')
        # Treat the leap year case
        elif np.shape(mvar)[0]==366:
            if percentile > 50:
                boolean = mvar > thresh_366
            elif percentile < 50:
                boolean = mvar < thresh_366
            intensity = np.asarray(mvar - thresh_366,dtype='float16')
            cat_index = np.asarray((mvar - clim_366)/normalizer_366,dtype='float16')
        # put into list
        print('put into list')
        all_boolean_list.append(boolean)
        all_intensity_list.append(intensity)
        all_cat_list.append(cat_index)
    # Now concatenate all arrays in the boolean array list and the cat index array list
    print('concatenate list')
    all_boolean_list = np.concatenate(all_boolean_list,axis=0)
    all_intensity_list = np.concatenate(all_intensity_list,axis=0)
    all_cat_list = np.concatenate(all_cat_list,axis=0)
    # make sure that the boolean and half precision is specified
    print('specify datatype')
    boolean_all = np.array(all_boolean_list,dtype='bool')
    intensity_all = np.array(all_intensity_list,dtype='float32')
    cat_index_all = np.array(all_cat_list,dtype='float32')
    print('done')
    return boolean_all, intensity_all, cat_index_all, mfn

def generate_boolean_smoothing_kernel(dimension_name='temporal'):#params,dimension_name='temporal'):
    if dimension_name == 'horizontal':     
        boolean_smoothing_kernel = np.ones((3,3,3,3))
        boolean_smoothing_kernel[:,[0,2],:,:] = False         ## turn of vertical connection
        boolean_smoothing_kernel[[0,2],:,:,:] = False         ## turn of temporal connection
    elif dimension_name == 'vertical':     
        boolean_smoothing_kernel = np.zeros((3,3,3,3))
        boolean_smoothing_kernel[1,:,1,1]=True                ## turn on vertical connection
    elif dimension_name == 'temporal':
        boolean_smoothing_kernel = np.zeros((3,3,3,3))
        boolean_smoothing_kernel[:,1,1,1]=True                ## turn on temporal connection
    return boolean_smoothing_kernel 

def pad_for_initial_closing(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def pad_for_initial_opening(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 1)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def smooth_boolean_array(params,unsmoothed_array,boolsmoother='hobday2016'):
    # run the smoothing commands depending on the employed methodology
    print('Do morphological operations.')
    print('Smooth the boolean according to {}.'.format(boolsmoother))
    #
    if boolsmoother == 'koehn2024':
        smoothing_kernel = generate_boolean_smoothing_kernel(dimension_name='temporal')
        closing_iterations = 2
        opening_iterations = 2
        padder = 2
        unsmoothed_array_padded = np.pad(unsmoothed_array,padder,pad_for_initial_closing)
        closed_array_padded = scipy.ndimage.binary_closing(unsmoothed_array_padded,structure=smoothing_kernel,iterations=closing_iterations)
        smoothed_array_padded = scipy.ndimage.binary_opening(closed_array_padded,structure=smoothing_kernel,iterations=opening_iterations)
        smoothed_array = smoothed_array_padded[padder:-padder,padder:-padder,padder:-padder,padder:-padder]
    elif boolsmoother == 'hobday2016':
        smoothing_kernel = generate_boolean_smoothing_kernel(dimension_name='temporal')
        opening_iterations = 2
        closing_iterations = 1
        padder = 2
        unsmoothed_array_padded = np.pad(unsmoothed_array,padder,pad_for_initial_opening)
        opened_array_padded = scipy.ndimage.binary_opening(unsmoothed_array_padded,structure=smoothing_kernel,iterations=opening_iterations)
        smoothed_array_padded = scipy.ndimage.binary_closing(opened_array_padded,structure=smoothing_kernel,iterations=closing_iterations)
        smoothed_array = smoothed_array_padded[padder:-padder,padder:-padder,padder:-padder,padder:-padder]
    elif boolsmoother == 'no_smoothing':
        smoothed_array = unsmoothed_array
    else:
        raise ValueError('boolsmoother is not well defined.')   
    return np.array(smoothed_array,dtype='bool')

def generate_xarray_dataset(params,boolean_raw,boolean_smoothed_hobday2016,boolean_smoothed_koehn2024,intensity_all,cat_index_all,mfn):
    # put everything into one dictionary
    boolean_dict = dict()
    boolean_dict['depth'] = {"dims": ("depth"), "data": mfn.depth.values,'attrs': {'units':'m'}}
    boolean_dict['time'] = {"dims": ("time"), "data": np.arange(np.shape(cat_index_all)[0]), 'attrs': {'units':'day of year'}}
    boolean_dict['lat'] = {"dims": ("latitude","longitude"), "data": mfn.lat_rho.values, 'attrs': {'units':'degrees N'}}
    boolean_dict['lon'] = {"dims": ("latitude","longitude"), "data": mfn.lon_rho.values, 'attrs': {'units':'degrees E'}}
    boolean_dict['boolean_raw'] = {"dims": ("time", "depth", "lat", "lon"), "data": boolean_raw, 'attrs': {'units':'True - extreme, False - not extreme'}}
    boolean_dict['boolean_smoothed_hobday2016'] = {"dims": ("time", "depth", "lat", "lon"), "data": boolean_smoothed_hobday2016, 'attrs': {'units':'True - extreme, False - not extreme'}}
    boolean_dict['boolean_smoothed_koehn2024'] = {"dims": ("time", "depth", "lat", "lon"), "data": boolean_smoothed_koehn2024, 'attrs': {'units':'True - extreme, False - not extreme'}}    
    boolean_dict['intensity_abs'] = {"dims": ("time", "depth", "lat", "lon"), "data": intensity_all, 'attrs': {'units':'°C (T minus thresh)'}}
    boolean_dict['cat_index_smoothed'] = {"dims": ("time", "depth", "lat", "lon"), "data": cat_index_all, 'attrs': {'units':'None ("magnitude" (T minus clim) normalized with normalizer (thresh-clim)'}}
    
    # make xarray dataset from dictionary
    ds = xr.Dataset.from_dict(boolean_dict)
    
    # add the attributes
    ds.attrs['author'] = 'E. E. Koehn'
    ds.attrs['date'] = str(date.today())
    ds.attrs['scriptdir'] = scriptdir
    ds.attrs['scriptname'] = scriptname
    ds.attrs['casename'] = casename
    ds.attrs['case_description'] = 'Given by the following attributes:'
    # for pattribute in dir(params):
    #     if 'labeled' and 'keep_only' not in pattribute:
    #         if not pattribute.startswith('__') and not callable(getattr(params,pattribute)):
    #             print(pattribute)
    #             print(eval('params.'+pattribute))
    #             ds.attrs['params.'+pattribute] = eval('params.'+pattribute)
    #         if not pattribute.startswith('__') and callable(getattr(params,pattribute)):
    #             print(pattribute)
    #             print(eval('params.'+pattribute+'()'))
    #             ds.attrs['params.'+pattribute+'()'] = eval('params.'+pattribute+'()')
    return ds

def save_to_netcdf(params,ds,config,scenario,varia,percentile):
    print("Saving")
    # save netcdf file
    if not os.path.exists('{}{}'.format(params.dir_root_boolean_arrays,config)):
        os.mkdir('{}{}'.format(params.dir_root_boolean_arrays,config))
        print('{}{} created. '.format(params.dir_root_boolean_arrays,config))
    else:
        print('{}{} exists already. Do nothing. '.format(params.dir_root_boolean_arrays,config))
    if not os.path.exists('{}{}/{}'.format(params.dir_root_boolean_arrays,config,scenario)):
        os.mkdir('{}{}/{}'.format(params.dir_root_boolean_arrays,config,scenario))
        print('{}{}/{} created.'.format(params.dir_root_boolean_arrays,config,scenario))
    else:
        print('{}{}/{} exists already. Do nothing.'.format(params.dir_root_boolean_arrays,config,scenario))
    savepath = '/nfs/sea/work/bblaser/data/temp/sliced/' #'{}{}/{}/'.format(params.dir_root_boolean_arrays,config,scenario)
    #savepath = params.dir_boolean_arrays
    netcdfname = params._fname_boolean_array(varia)
    netcdfname = netcdfname.replace(f'{params.threshold_percentile}perc',f'{percentile}perc')
    print(netcdfname)
    ds.to_netcdf(savepath+netcdfname)
    print("Saving done")


#%% DEFINE CASE AND START THE CALCULATION
#casedir = '/home/koehne/Documents/publications/paper_future_simulations/scripts/modules/cases/'
casedir = '/home/bblaser/scripts/bsc_beni/eike_reference/modules/cases/'    # Define directory containing different yaml files
casenames = ['case00.yaml'] 

config = 'romsoc_fully_coupled' #'roms_only' #'romsoc_fully_coupled' #'romsoc_fully_coupled'
scenario = 'present' #'ssp585'
varia = 'temp' # 'temp'

for casename in casenames:
    params = read_config_files(casedir+casename)

    root_percentile = params.threshold_percentile
    if varia == 'O2':
        percentile = 100 - root_percentile
    elif varia == 'temp' or varia == 'Hions':
        percentile = root_percentile

    #%%
    thresh_365, clim_365, normalizer_365 = load_threshold_and_climatology(params,config,scenario,varia,percentile)
    thresh_366, clim_366, normalizer_366 = calculate_366_arrays(thresh_365, clim_365, normalizer_365)
    #%%
    boolean_raw, intensity_all, cat_index_all, mfn = generate_and_concatenate_annual_boolean_files(params, thresh_365, clim_365, normalizer_365, thresh_366, clim_366, normalizer_366,config,scenario,varia,percentile)
    #%%
    boolean_smoothed_hobday2016 = smooth_boolean_array(params,boolean_raw,boolsmoother='hobday2016')
    boolean_smoothed_koehn2024 = smooth_boolean_array(params,boolean_raw,boolsmoother='koehn2024')
    ds = generate_xarray_dataset(params, boolean_raw, boolean_smoothed_hobday2016, boolean_smoothed_koehn2024, intensity_all, cat_index_all, mfn)
    #%%
    save_to_netcdf(params,ds,config,scenario,varia,percentile)
# %%
