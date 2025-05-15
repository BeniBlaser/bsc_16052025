"""
author: Eike E. Köhn (modified by GitHub Copilot)
date: April 10, 2025
description: This file takes ROMS/ROMSOC ocean model output and calculates climatologies as defined in the case-*.yaml file.
"""

#%% 
print('Define scriptname and scriptdir.')
scriptdir = '/home/bblaser/scripts/bsc_beni/eike_reference_copy/detect_extreme_events/'
scriptname = 'calc_climatology.py'

#%% 
import sys
import os

# Add the modules directory to the path
sys.path.insert(0, '/home/bblaser/scripts/bsc_beni/eike_reference_copy/modules')

# Import specific modules you need directly
import numpy as np
import xarray as xr
import scipy.ndimage
from datetime import date
# Add other imports you need

exec(open('/home/bblaser/scripts/bsc_beni/eike_reference_copy/modules/define_cases_and_parameters.py').read())

#%% DEFINE THE CORE FUNCTIONS
def get_shape_of_dummy_input_file(parameters, config, scenario, varia):
    # Use the directory from parameters, but adapt to the file naming pattern we see
    dummy_file_dir = '/nfs/sea/work/bblaser/z_avg_meanpool/'
    
    # Build file name using the pattern z_avg_YYYY_MMM.nc
    if config == 'roms_only':
        dummy_year = str(parameters.threshold_period_start_year)
        # Using month 001 as example
        dummy_file_fname = f"z_avg_{dummy_year}_001.nc"
        dummy_file = dummy_file_dir + dummy_file_fname
        
    elif config == 'romsoc_fully_coupled':
        dummy_year = str(parameters.threshold_period_start_year)
        dummy_file_fname = f"z_avg_{dummy_year}_001_37zlevs_full_1x1meanpool_downsampling.nc"
        dummy_file = dummy_file_dir + dummy_file_fname
    
    try: 
        dummyfn = xr.open_dataset(dummy_file)
        print(f"Available variables in {dummy_file}:")
        print(list(dummyfn.variables))
    except Exception as e:
        print(f"Error opening file {dummy_file}: {e}")
        raise FileNotFoundError(f'No dummy file found at {dummy_file}')
        
    # Check if the variable name matches something in the file
    if varia == 'Hions':
        dummy_shape = dummyfn.variables['pH_offl'].shape
    elif varia == 'TOT_PROD' or varia == 'POC_FLUX_IN':
        prod_vars = [var for var in dummyfn.variables if 'prod' in var.lower()]
        
        if prod_vars:
            prod_var_name = prod_vars[0]
            dummy_shape = dummyfn.variables[prod_var_name].shape
        else:
            try:
                dummy_shape = dummyfn.variables[varia].shape
            except KeyError:
                print(f"ERROR: Could not find {varia} variable or similar.")
                print("Available variables:", list(dummyfn.variables))
                raise KeyError(f"{varia} variable not found in the dataset.")
    else:
        dummy_shape = dummyfn.variables[varia].shape
            
    return dummy_shape

def generate_dummy_day_list(doy, parameters):
    # generate list of days to load
    half_window = (parameters.threshold_aggregation_window_size - 1)/2.
    dummy_day_list = np.arange(doy-half_window, doy+half_window+1, dtype='int32')
    dummy_day_list[dummy_day_list<0] = 365+dummy_day_list[dummy_day_list<0]
    dummy_day_list[dummy_day_list>364] = dummy_day_list[dummy_day_list>364]-365 
    return dummy_day_list

def retrieve_model_file(parameters, year, config, scenario, month=None):
    model_file_dir = '/nfs/sea/work/bblaser/z_avg_meanpool/'
    
    if month is None:
        # Return all months as before
        if config == 'roms_only':
            return_val = [f"{model_file_dir}z_avg_{year}_{m:03d}.nc" for m in range(1, 13)]  # Changed from 12 to 13
        elif config == 'romsoc_fully_coupled':
            return_val = [f"{model_file_dir}z_avg_{year}_{m:03d}_37zlevs_full_1x1meanpool_downsampling.nc" for m in range(1, 13)]  # Changed from 12 to 13
    else:
        # Return just one month
        if config == 'roms_only':
            return_val = f"{model_file_dir}z_avg_{year}_{month:03d}.nc"
        elif config == 'romsoc_fully_coupled':
            return_val = f"{model_file_dir}z_avg_{year}_{month:03d}_37zlevs_full_1x1meanpool_downsampling.nc"
    
    return return_val

def open_yearfile(model_file, config):
    if config == 'roms_only':
        filehandle = xr.open_dataset(model_file)
    elif config == 'romsoc_fully_coupled':
        filehandle = xr.open_mfdataset(model_file, concat_dim='time', combine='nested', parallel=True, data_vars='minimal')
    return filehandle

def adjust_daylist_for_leapyears(dummy_day_list, year):
    # Adjust the daylist for leap years, i.e. skip the 29th of february
    if np.mod(year, 4) == 0:
        dummy_day_list[dummy_day_list>59] = dummy_day_list[dummy_day_list>59]+1
        daylist = list(dummy_day_list)
    else:
        daylist = list(dummy_day_list)
    return daylist

def set_up_arrays(dummy_shape, days_in_year=365):
    # Extract all dimensions except the first (time dimension)
    remaining_dims = dummy_shape[1:]
    
    # Create new array with 365 days but keeping all other dimensions
    new_shape = (days_in_year,) + remaining_dims
    
    climatology_array = np.zeros(new_shape)
    return climatology_array

def calculate_climatology(params, config, scenario, varia):
    dummy_shape = get_shape_of_dummy_input_file(params, config, scenario, varia)
    
    numyears = params.threshold_period_end_year - params.threshold_period_start_year + 1
    climatology_array = set_up_arrays(dummy_shape, params.threshold_daysinyear)
    half_window = int((params.threshold_aggregation_window_size - 1) / 2.)
    
    # Loop through days of the year
    for didx, doy in enumerate(range(params.threshold_daysinyear)):
        print(f"Processing day {doy} of {params.threshold_daysinyear}")
        dataarray = np.zeros((numyears, params.threshold_aggregation_window_size,
                             dummy_shape[-3], dummy_shape[-2], dummy_shape[-1]))
        
        # Loop through years to load data
        for yidx, year in enumerate(range(params.threshold_period_start_year, params.threshold_period_end_year + 1)):
            dummy_day_list = generate_dummy_day_list(doy, params)
            model_file = retrieve_model_file(params, year, config, scenario)
            fn = open_yearfile(model_file, config)
            daylist = adjust_daylist_for_leapyears(dummy_day_list, year)
            
            if varia == 'Hions':
                dummy = fn.variables['pH_offl'][daylist, ...].values
                dataarray[yidx, ...] = 10 ** (-1 * dummy)
            elif varia == 'TOT_PROD' or varia == 'POC_FLUX_IN':
                prod_vars = [var for var in fn.variables if 'prod' in var.lower()]
                if prod_vars:
                    prod_var_name = prod_vars[0]
                    dataarray[yidx, ...] = fn.variables[prod_var_name][daylist, ...].values      
                else:
                    try:
                        dataarray[yidx, ...] = fn.variables[varia][daylist, ...].values
                    except KeyError:
                        print(f"ERROR: Could not find {varia} variable or similar.")
                        print("Available variables:", list(fn.variables))
                        raise KeyError(f"{varia} variable not found in the dataset.")
            else:
                dataarray[yidx, ...] = fn.variables[varia][daylist, ...].values
        
        # Calculate climatology for each depth level
        for k in range(dummy_shape[-3]):
            if np.mod(k, 10) == 0:
                print(f"  Depth level {k}")
            # Calculate climatology by averaging across years for the central day in the window
            climdum = np.mean(dataarray[:, half_window, k, ...], axis=0)
            climatology_array[doy, k, ...] = climdum
        
        del dataarray
    
    return climatology_array, fn

def smoothing_climatology(params, climatology_array, config, scenario, varia):
    print("Smoothing climatology...")
    dummy_shape = get_shape_of_dummy_input_file(params, config, scenario, varia)
    kernel = np.ones(params.threshold_smoothing_window_size)
    smoothed_climatology = np.zeros_like(climatology_array)
    
    for k in range(dummy_shape[-3]):
        smoothed_climatology[:,k,:,:] = scipy.ndimage.convolve1d(
            climatology_array[:,k,:,:], 
            kernel, 
            axis=0, 
            mode='wrap'
        ) / params.threshold_smoothing_window_size
        
    return smoothed_climatology

def save_climatology(params, fn, climatology_array_smoothed, config, scenario, varia):
    print("Preparing to save climatology data...")
    clim_dict = dict()
    
    # Define dimensions
    clim_dict['depth'] = {"dims": ("depth"), "data": fn.depth.values, 'attrs': {'units':'m'}}
    clim_dict['time'] = {"dims": ("time"), "data": np.arange(params.threshold_daysinyear), 'attrs': {'units':'day of year'}}
    clim_dict['lat'] = {"dims": ("latitude","longitude"), "data": fn.lat_rho.values, 'attrs': {'units':'degrees N'}}
    clim_dict['lon'] = {"dims": ("latitude","longitude"), "data": fn.lon_rho.values, 'attrs': {'units':'degrees E'}}
    
    # Set appropriate units
    if varia == 'temp':
        unit = '°C'
    elif varia == 'O2':
        unit = 'mmol m-3'
    elif varia == 'TOT_PROD':
        unit = 'mmol/m3/s'
    elif varia == 'Hions':
        unit = 'mol L-1'
    elif varia == 'POC_FLUX_IN':
        unit = 'mmol C/m2/s'
    else:
        unit = 'not specified'
    
    # Add climatology data
    clim_dict['climatology'] = {
        "dims": ("time", "depth", "lat", "lon"), 
        "data": climatology_array_smoothed.astype(np.float32), 
        'attrs': {'units': unit}
    }
    
    # Create dataset
    ds = xr.Dataset.from_dict(clim_dict)
    
    # Add metadata
    ds.attrs['author'] = 'E. E. Koehn (modified script)'
    ds.attrs['date'] = str(date.today())
    ds.attrs['scriptdir'] = scriptdir
    ds.attrs['scriptname'] = scriptname
    ds.attrs['casename'] = casename
    ds.attrs['description'] = f'Daily climatology for {varia}'
    ds.attrs['leap_years'] = 'Feb 29th values were discarded for the computation of climatologies. They are thus 365 days long. In order to get a 366th value for leap years, I propose to linearly interpolate the values between Feb 28 and Mar 1.'
    
    # Create directories if needed
    savepath = '/nfs/sea/work/bblaser/data/climatology/' 
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print(f'Created directory: {savepath}')
        
    if not os.path.exists(f'{savepath}{config}'):
        os.mkdir(f'{savepath}{config}')
        print(f'Created directory: {savepath}{config}')
    
    if not os.path.exists(f'{savepath}{config}/{scenario}'):
        os.mkdir(f'{savepath}{config}/{scenario}')
        print(f'Created directory: {savepath}{config}/{scenario}')
    
    # Generate filename and save
    save_filename = f"climatology_{varia}_{config}_{scenario}.nc"
    ds.to_netcdf(f'{savepath}{config}/{scenario}/{save_filename}')
    print(f"Climatology saved to {savepath}{config}/{scenario}/{save_filename}")

#%% DEFINE PARAMETERS AND FILENAMES ETC
casedir = '/home/bblaser/scripts/bsc_beni/eike_reference_copy/modules/cases/'
casenames = ['case00_TOT_PROD.yaml']    # 'case00_temp.yaml', 'case00_POC_FLUX_IN.yaml']

#%% RUN THE CLIMATOLOGY CALCULATION
config = 'romsoc_fully_coupled'
scenario = 'present'
varia = 'TOT_PROD'

for casename in casenames:
    print(f"Processing {casename}...")
    params = read_config_files(casedir+casename)
    
    print(f"Calculating climatology for {varia}...")
    climatology_array, fn = calculate_climatology(params, config, scenario, varia)
    
    print("Smoothing climatology...")
    climatology_array_smoothed = smoothing_climatology(params, climatology_array, config, scenario, varia)
    
    print("Saving results...")
    save_climatology(params, fn, climatology_array_smoothed, config, scenario, varia)
    
    print("Processing complete!")

# %%