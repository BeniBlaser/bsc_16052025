"""
author: Eike E. Köhn
date: Jan 18, 2024
description: This file takes ROMS/ROMSOC ocean model output and calculates the climatologies and thresholds as defined in the case-*.yaml file and following the approach of Hobday et al. 2016.
"""

#%% 
print('Define scriptname and scriptdir.')
scriptdir = '/home/koehne/Documents/publications/paper_future_simulations/scripts/detect_extreme_events/'
scriptname = 'calc_threshold_and_climatology.py'

#%% 
print('Load packages and functions.')
exec(open('../modules/import_package_selection.py').read())
exec(open('../modules/define_cases_and_parameters.py').read())
import scipy.ndimage

#%% DEFINE THE CORE FUNCTIONS
def get_shape_of_dummy_input_file(parameters,config,scenario,varia):
    dummy_file_dir = eval('parameters.dir_{}_{}_daily_zlevs'.format(config,scenario))
    if config == 'roms_only':
        dummy_file_fname = parameters._fname_model_output_regridded().replace('YYYY',str(parameters.threshold_period_start_year))
        dummy_file = dummy_file_dir+dummy_file_fname
        try: 
            dummyfn = xr.open_dataset(dummy_file)
        except:
            raise FileNotFoundError('No dummy file to be found.')
        if varia == 'Hions':
            dummy_shape = dummyfn.variables['pH_offl'].shape
        else:
            dummy_shape = dummyfn.variables[varia].shape
    elif config == 'romsoc_fully_coupled':
        dummy_file_fname = parameters._fname_model_output_regridded_romsoc().replace('YYYY',str(parameters.threshold_period_start_year))
        dummy_files = [dummy_file_dir+dummy_file_fname.replace('MMM','{:03d}'.format(int(mon+1))) for mon in range(12)]
        try: 
            dummyfn = xr.open_mfdataset(dummy_files,concat_dim='time',combine='nested',parallel=True,data_vars='minimal')
        except:
            raise FileNotFoundError('No dummy file to be found.')
        if varia == 'Hions':
            dummy_shape = dummyfn.variables['pH_offl'].shape
        else:
            dummy_shape = dummyfn.variables[varia].shape
    return dummy_shape

def generate_dummy_day_list(doy,parameters):
    # generate list of days to load
    half_window = (parameters.threshold_aggregation_window_size - 1)/2.
    dummy_day_list = np.arange(doy-half_window,doy+half_window+1,dtype='int32')
    dummy_day_list[dummy_day_list<0]=365+dummy_day_list[dummy_day_list<0]
    dummy_day_list[dummy_day_list>364]=dummy_day_list[dummy_day_list>364]-365 
    return dummy_day_list

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

def adjust_daylist_for_leapyears(dummy_day_list,year):
    # Adjust the daylist for leap years, i.e. skip the 29th of february
    if np.mod(year,4)==0:
        dummy_day_list[dummy_day_list>59]= dummy_day_list[dummy_day_list>59]+1
        daylist = list(dummy_day_list)
    else:
        daylist = list(dummy_day_list)
    return daylist

def set_up_arrays(dummy_shape):
    array1 = np.zeros(dummy_shape)
    array2 = np.zeros(dummy_shape)
    array3 = np.zeros(dummy_shape)
    return array1, array2, array3

def core_calcuation(params,config,scenario,varia,percentile):
    dummy_shape = get_shape_of_dummy_input_file(params,config,scenario,varia)
    numyears = params.threshold_period_end_year-params.threshold_period_start_year+1
    threshold_array,climatology_array,intensity_normalizer_array = set_up_arrays(dummy_shape)
    half_window = int((params.threshold_aggregation_window_size - 1)/2.)
    # now loop through days of the year
    for didx,doy in enumerate(range(params.threshold_daysinyear)):
        print(doy)
        dataarray = np.zeros((numyears,params.threshold_aggregation_window_size,dummy_shape[-3],dummy_shape[-2],dummy_shape[-1]))
        # loop through years to load data
        for yidx,year in enumerate(range(params.threshold_period_start_year,params.threshold_period_end_year+1)):
            #print(doy,year)
            dummy_day_list = generate_dummy_day_list(doy,params)
            model_file = retrieve_model_file(params,year,config,scenario)
            #print(model_file)
            fn = open_yearfile(model_file,config)
            #print(fn)
            daylist = adjust_daylist_for_leapyears(dummy_day_list,year)
            if varia == 'Hions':
                dummy = fn.variables['pH_offl'][daylist,...].values
                dataarray[yidx,...] = 10**(-1*dummy)
            else:
                dataarray[yidx,...] = fn.variables[varia][daylist,...].values
        # calculate the percentile across the first two dimensions    
        for k in range(dummy_shape[-3]):
            if np.mod(k,10)==0:
                print(doy,k)
            threshdum = np.percentile(dataarray[:,:,k,...],percentile,axis=(0,1))
            threshold_array[doy,k,...] = threshdum
            climdum = np.mean(dataarray[:,half_window,k,...],axis=0)
            climatology_array[doy,k,...] = climdum
        del dataarray
    intensity_normalizer_array = threshold_array - climatology_array
    return threshold_array, climatology_array, intensity_normalizer_array, fn

def smoothing(params,threshold_array,climatology_array,config,scenario,varia):
    print("Smoothing")
    dummy_shape = get_shape_of_dummy_input_file(params,config,scenario,varia)
    kernel = np.ones(params.threshold_smoothing_window_size)
    smoothed_threshold = np.zeros_like(threshold_array)
    smoothed_climatology = np.zeros_like(climatology_array)
    for k in range(dummy_shape[-3]):
        smoothed_threshold[:,k,:,:] = scipy.ndimage.convolve1d(threshold_array[:,k,:,:],kernel,axis=0,mode='wrap')/params.threshold_smoothing_window_size
        smoothed_climatology[:,k,:,:] = scipy.ndimage.convolve1d(climatology_array[:,k,:,:],kernel,axis=0,mode='wrap')/params.threshold_smoothing_window_size
    smoothed_intensity_normalizer = smoothed_threshold-smoothed_climatology
    return smoothed_threshold, smoothed_climatology, smoothed_intensity_normalizer

def saving(params,fn,threshold_array_smoothed,climatology_array_smoothed,intensity_normalizer_array_smoothed,config,scenario,varia,percentile):
    thresh_clim_dict = dict()
    thresh_clim_dict['depth'] = {"dims": ("depth"), "data": fn.depth.values,'attrs': {'units':'m'}}
    thresh_clim_dict['time'] = {"dims": ("time"), "data": np.arange(params.threshold_daysinyear), 'attrs': {'units':'day of year'}}
    thresh_clim_dict['lat'] = {"dims": ("latitude","longitude"), "data": fn.lat_rho.values, 'attrs': {'units':'degrees N'}}
    thresh_clim_dict['lon'] = {"dims": ("latitude","longitude"), "data": fn.lon_rho.values, 'attrs': {'units':'degrees E'}}
    #thresh_clim_dict['thresh_raw'] = {"dims": ("time", "depth", "lat", "lon"), "data": threshold_array.astype(np.float32), 'attrs': {'units':'°C'}}
    #thresh_clim_dict['clim_raw'] = {"dims": ("time", "depth", "lat", "lon"), "data": climatology_array.astype(np.float32), 'attrs': {'units':'°C'}}
    #thresh_clim_dict['intensity_normalizer_raw'] = {"dims": ("time", "depth", "lat", "lon"), "data": intensity_normalizer_array.astype(np.float32), 'attrs': {'units':'°C'}}
    if varia == 'temp':
        unit = '°C'
    elif varia == 'O2':
        unit = 'mmol m-3'
    elif varia == 'Hions':
        unit = 'mol L-1'
    else:
        unit = 'not specified'
    thresh_clim_dict['thresh_smoothed'] = {"dims": ("time", "depth", "lat", "lon"), "data": threshold_array_smoothed.astype(np.float32), 'attrs': {'units':unit}}
    thresh_clim_dict['clim_smoothed'] = {"dims": ("time", "depth", "lat", "lon"), "data": climatology_array_smoothed.astype(np.float32), 'attrs': {'units':unit}}
    thresh_clim_dict['intensity_normalizer_smoothed'] = {"dims": ("time", "depth", "lat", "lon"), "data": intensity_normalizer_array_smoothed.astype(np.float32), 'attrs': {'units':unit}}
    #thresh_clim_dict['params'] = params
    # make netcdf file from dictionary
    ds = xr.Dataset.from_dict(thresh_clim_dict)
    # add the attributes
    ds.attrs['author'] = 'E. E. Koehn'
    ds.attrs['date'] = str(date.today())
    ds.attrs['scriptdir'] = scriptdir
    ds.attrs['scriptname'] = scriptname
    ds.attrs['casename'] = casename
    ds.attrs['case_description'] = 'Given by the following attributes:'
    #for pattribute in dir(params):
    #    if 'boolean' not in pattribute and 'keep' not in pattribute and 'labeled' not in pattribute:
    #        if not pattribute.startswith('__') and not callable(getattr(params,pattribute)):
    #            print(pattribute)
    #            print(eval('params.'+pattribute))
    #            ds.attrs['params.'+pattribute] = eval('params.'+pattribute)
    #        if not pattribute.startswith('__') and callable(getattr(params,pattribute)):
    #            print(pattribute)
    #            print(eval('params.'+pattribute+'()'))
    #            ds.attrs['params.'+pattribute+'()'] = eval('params.'+pattribute+'()')
    ds.attrs['leap_years'] = 'Feb 29th values were discarded for the computation of the threshold and climatologies. They are thus 365 days long. In order to get a 366th value for leap years, I propose to linearly interpolate the values between Feb 28 and Mar 1.'
    # save netcdf file
    print("Saving")
    if not os.path.exists('{}{}'.format(params.dir_root_thresholds_and_climatologies,config)):
        os.mkdir('{}{}'.format(params.dir_root_thresholds_and_climatologies,config))
        print('{}{} created. '.format(params.dir_root_thresholds_and_climatologies,config))
    else:
        print('{}{} exists already. Do nothing. '.format(params.dir_root_thresholds_and_climatologies,config))
    if not os.path.exists('{}{}/{}'.format(params.dir_root_thresholds_and_climatologies,config,scenario)):
        os.mkdir('{}{}/{}'.format(params.dir_root_thresholds_and_climatologies,config,scenario))
        print('{}{}/{} created.'.format(params.dir_root_thresholds_and_climatologies,config,scenario))
    else:
        print('{}{}/{} exists already. Do nothing.'.format(params.dir_root_thresholds_and_climatologies,config,scenario))
    
    # Force use of the temp directory
    temp_path = '/nfs/sea/work/bblaser/data/temp/'
    print(f"FORCING SAVE TO TEMP PATH: {temp_path}")
    
    save_filename = params._fname_threshold_and_climatology(varia).replace(str(params.threshold_percentile)+'perc',str(int(percentile))+'perc')
    print(f"SAVING TO: {temp_path + save_filename}")
    
    # Save to temp directory
    ds.to_netcdf(temp_path + save_filename)
    print(f"File saved to: {temp_path + save_filename}")

def calc_threshold_array(params):
    threshold_array,          climatology_array,          intensity_normalizer_array, fn      = core_calcuation(params)
    threshold_array_smoothed, climatology_array_smoothed, intensity_normalizer_array_smoothed = smoothing(params,threshold_array,climatology_array,config,scenario,varia)
    saving(params,fn,threshold_array_smoothed,climatology_array_smoothed,intensity_normalizer_array_smoothed,config,scenario,varia)

#%% DEFINE PARAMETERS AND FILENAMES ETC
casedir = '/home/bblaser/scripts/bsc_beni/eike_reference/modules/cases/'    # Define directory containing different yaml files
casenames = ['case00_80th.yaml']                      # Define list of cases for which to do the regridding

#%% RUN THE THRESHOLD CALCULATION
config = 'romsoc_fully_coupled' #'roms_only' 
scenario = 'present'#'present'#'present'
varia = 'temp'#'omega_arag_offl'#'Hions'

for casename in casenames:
    params = read_config_files(casedir+casename)                                        # Load in parameters
    root_percentile = params.threshold_percentile
    if varia == 'O2' or varia == 'omega_arag_offl':
        percentile = 100 - root_percentile
    elif varia == 'temp' or varia == 'Hions':
        percentile = root_percentile
    
    # Print message before calculations start
    temp_path = '/nfs/sea/work/bblaser/data/temp/'
    print(f"\n=== STARTING CALCULATIONS ===")
    print(f"Results will be saved to: {temp_path}")
    print(f"Variable: {varia}, Percentile: {percentile}")
    print(f"===========================\n")
    
    #calc_threshold_array(params)                                                       # launch the threshold calulation
    threshold_array,          climatology_array,          intensity_normalizer_array, fn      = core_calcuation(params,config,scenario,varia,percentile)
    threshold_array_smoothed, climatology_array_smoothed, intensity_normalizer_array_smoothed = smoothing(params,threshold_array,climatology_array,config,scenario,varia)
    saving(params,fn,threshold_array_smoothed,climatology_array_smoothed,intensity_normalizer_array_smoothed,config,scenario,varia,percentile)

# %%
