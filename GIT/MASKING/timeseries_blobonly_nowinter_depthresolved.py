#!/usr/bin/env python3
"""
Create a comprehensive timeseries analysis spanning 2011-2021, including:
1. Pre-Blob (2011-2013): Whole domain statistics
2. Blob (2014-2016): MHW contour mask statistics
3. Post-Blob (2017-2021): Whole domain statistics

Each period is compared to the appropriate climatology:
- Whole domain periods (2011-2013, 2017-2021) use whole domain climatology
- MHW periods (2014-2016) use the same MHW mask applied to climatology years

IMPORTANT: This version only includes months April (003) through November (011)
"""
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean as cmo
import os
import glob
import re
from pathlib import Path
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.table as tbl

# %%
#--------- CONFIGURATION ---------#
# Paths
DATA_DIR = '/nfs/sea/work/bblaser/z_avg_meanpool_domain/'
OUTPUT_DIR = '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/complete_timeseries_apr_nov/'
BOOLEAN_FILE_PATH = '/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/extreme_analysis/boolean_arrays/romsoc_fully_coupled/present/boolean_array_hobday2016_threshold_and_climatology_temp_37zlevs_full_1x1meanpool_downsampling_2011-2021analysisperiod_90perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing.nc' #'/nfs/sea/work/bblaser/data/temp/boolean_array_hobday2016_threshold_and_climatology_temp_37zlevs_full_1x1meanpool_downsampling_2011-2021analysisperiod_95perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing.nc'
BOOL_VAR_NAME = 'boolean_smoothed_hobday2016'

# Study domain boundaries
STUDY_REGION = {
    "lon_min": 205, "lon_max": 245,  # Full longitude range
    "lat_min": 10, "lat_max": 62,    # Full latitude range
    "name": "Northeast Pacific"
}

# Time periods
PRE_BLOB_YEARS = range(2011, 2014)  # 2011-2013
BLOB_YEARS = range(2014, 2017)      # 2014-2016
POST_BLOB_YEARS = range(2017, 2022) # 2017-2021
ALL_YEARS = range(2011, 2022)       # All years for climatology

# Months to include in analysis (April-November)
ANALYSIS_MONTHS = range(5, 11)  # 4-11 correspond to Apr-Nov in the data files

# Variables to analyze
VARIABLES = [
    {
        'name': 'POC_FLUX_IN',
        'type': 'single_depth',
        'target_depth': -100,
        'cmap': 'dense',
        'title': 'POC Flux at 100m'
    },
    {
        'name': 'TOT_PROD',
        'type': 'depth_averaged',
        'depth_levels': [0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50],
        'cmap': 'tempo',
        'title': 'Total Productivity'
    },
    {
        'name': 'temp',
        'type': 'depth_averaged',
        'depth_levels': [0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50],
        'cmap': 'thermal',
        'title': 'Temperature'
    }
]

# MHW frequency thresholds for blob period mask
MHW_THRESHOLD_MIN = [0.1, 0.5, 0.9]  # Lower bound for contour
MHW_THRESHOLD_TITLE = [int(threshold * 100) for threshold in MHW_THRESHOLD_MIN]  # Convert to percentages for titles
MHW_THRESHOLD_MAX = 1.0  # Upper bound for contour

# Include whole region calculation for comparison with MHW-masked anomalies
INCLUDE_WHOLE_REGION = True  # Set to True to include anomalies over entire study region

# Define conversion factors for units
UNIT_CONVERSION = {
    'temp': 1.0,  # No conversion needed
    'TOT_PROD': 31536.0,  # Convert mmol/m^3/s to mol/m^3/year
    'POC_FLUX_IN': 31536.0  # Convert mmolC/m^2/s to molC/m^2/year
}

# %%
#--------- HELPER FUNCTIONS ---------#
def get_month_doy_range(year, month):
    """Get the day of year range for a specific month in a specific year."""
    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days_per_month = [31, 29 if is_leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    doy_start = sum(days_per_month[:month-1]) + 1
    doy_end = doy_start + days_per_month[month-1] - 1
    return (doy_start, doy_end)

def load_boolean_data():
    """Load the boolean dataset with extreme events."""
    print("\nLoading boolean extreme event dataset...")
    try:
        bool_ds = xr.open_dataset(BOOLEAN_FILE_PATH)
        if BOOL_VAR_NAME not in bool_ds:
            print(f"ERROR: Variable {BOOL_VAR_NAME} not found in boolean dataset")
            print(f"Available variables: {list(bool_ds.data_vars)}")
            return None
        
        # Get the coordinates
        lons = bool_ds['lon'].values if 'lon' in bool_ds else None
        lats = bool_ds['lat'].values if 'lat' in bool_ds else None
        
        if lons is None or lats is None:
            print("WARNING: Geographic coordinates not found in boolean dataset")
            return None
            
        print(f"✓ Loaded boolean data with dimensions {bool_ds[BOOL_VAR_NAME].shape}")
        return bool_ds, lons, lats
        
    except Exception as e:
        print(f"Error loading boolean data: {e}")
        return None

def load_variable_data(year, month):
    """Load variable data for a specific month and year."""
    # Format month with leading zeros
    month_str = f"{month:03d}"
    
    # Create the file pattern
    file_pattern = os.path.join(DATA_DIR, f"z_avg_{year}_{month_str}_37zlevs_full_1x1meanpool_downsampling.nc")
    
    print(f"Looking for variable data file: {file_pattern}")
    matching_files = glob.glob(file_pattern)
    
    if not matching_files:
        print(f"No files found for year {year}, month {month}")
        return None
    
    print(f"Loading data from {os.path.basename(matching_files[0])}")
    
    try:
        ds = xr.open_dataset(matching_files[0])
        return ds
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_domain_mask(ds):
    """Create a mask for the full study domain."""
    # Create mask from study region boundaries
    lon_min, lon_max = STUDY_REGION['lon_min'], STUDY_REGION['lon_max']
    lat_min, lat_max = STUDY_REGION['lat_min'], STUDY_REGION['lat_max']
    
    # ROMS model uses lon_rho and lat_rho for coordinates
    lon_var = None
    lat_var = None
    
    # Check for common coordinate names in ROMS and other models
    lon_names = ['lon', 'longitude', 'lon_rho', 'x_rho', 'x']
    lat_names = ['lat', 'latitude', 'lat_rho', 'y_rho', 'y']
    
    # Find the coordinate variables
    for name in lon_names:
        if name in ds:
            lon_var = name
            break
    
    for name in lat_names:
        if name in ds:
            lat_var = name
            break
    
    if lon_var is None or lat_var is None:
        print("WARNING: No lon/lat coordinates found, using entire domain")
        print(f"Available coordinates in dataset: {list(ds.coords)}")
        print(f"Available variables in dataset: {list(ds.data_vars)}")
        
        # Create a mask of all True values
        for var_name in ds.data_vars:
            if hasattr(ds[var_name], 'dims') and 'depth' in ds[var_name].dims:
                # Get shape for first depth level
                if 'time' in ds[var_name].dims:
                    shape = ds[var_name].isel(depth=0, time=0).shape
                else:
                    shape = ds[var_name].isel(depth=0).shape
                return np.ones(shape, dtype=bool)
    
    print(f"Using coordinates: {lon_var} and {lat_var}")
    
    # Create mask using the found coordinate variables
    region_mask = ((ds[lon_var] >= lon_min) & (ds[lon_var] <= lon_max) & 
                 (ds[lat_var] >= lat_min) & (ds[lat_var] <= lat_max))
    
    # Convert to boolean array
    mask = region_mask.values.astype(bool)
    
    # Print mask statistics
    n_points = np.sum(mask)
    total_points = mask.size
    percent = (n_points / total_points) * 100
    
    print(f"Created domain mask with {n_points} points ({percent:.2f}% of domain)")
    
    return mask

def get_monthly_extreme_data(bool_ds, year, month, depth_idx=0):
    """
    Extract monthly extreme event frequency for a specific month of a year at a specific depth.
    Now supports extracting data at any depth level.
    """
    if 'time' not in bool_ds.dims:
        print("ERROR: No time dimension in boolean dataset")
        return None
    
    # Get the time values
    time_values = bool_ds['time'].values
    
    # Calculate the DOY range for this month
    doy_start, doy_end = get_month_doy_range(year, month)
    
    # Calculate the time index for this year
    year_offset = (year - 2011) * 365  # 2011 is the start year from filename
    if year > 2012:  # Add leap days after 2012
        year_offset += (year - 2012) // 4
    
    # Calculate indices for this month in this year
    month_indices = np.where((time_values >= year_offset + doy_start) & 
                           (time_values <= year_offset + doy_end))
    
    if len(month_indices[0]) == 0:
        print(f"No boolean data found for {month}/{year}")
        return None
    
    # Get data for this month at the specified depth
    monthly_data = bool_ds[BOOL_VAR_NAME].isel(depth=depth_idx, time=month_indices[0])
    
    # Calculate mean frequency for the month
    monthly_frequency = monthly_data.mean(dim='time')
    
    print(f"Extracted extreme event frequency for {year}-{month:02d} at depth index {depth_idx} with {len(month_indices[0])} time points")
    return monthly_frequency, month_indices[0]

def create_mhw_contour_mask(monthly_freq, min_threshold=0.4, max_threshold=1.0, target_shape=None):
    """Create a mask from the MHW frequency contour."""
    # Create mask where frequency is between thresholds
    mask = (monthly_freq >= min_threshold) & (monthly_freq <= max_threshold)
    
    # Convert to boolean array
    mask = mask.values.astype(bool)
    
    # Resize mask if target shape is provided
    if target_shape is not None and mask.shape != target_shape:
        print(f"Resizing mask from {mask.shape} to match target shape {target_shape}")
        
        # Create new mask with target shape
        new_mask = np.zeros(target_shape, dtype=bool)
        
        # Copy overlapping region
        min_rows = min(mask.shape[0], target_shape[0])
        min_cols = min(mask.shape[1], target_shape[1])
        new_mask[:min_rows, :min_cols] = mask[:min_rows, :min_cols]
        
        mask = new_mask
    
    # Print mask statistics
    n_points = np.sum(mask)
    total_points = mask.size
    percent = (n_points / total_points) * 100
    
    print(f"Created MHW contour mask with {n_points} points ({percent:.2f}% of domain)")
    
    return mask

def calculate_masked_variable_stats(variable_ds, variable_config, mask):
    """Calculate statistics for a variable within the masked region."""
    # Get variable details
    variable_name = variable_config['name']  # <-- Add this line
    variable_type = variable_config['type']
    
    # Check if variable exists in dataset
    if variable_name not in variable_ds:
        return None
    
    # Get the variable data
    var_data = variable_ds[variable_name]
    
    # Get time dimension size
    if hasattr(var_data, 'dims') and 'time' in var_data.dims:
        n_times = var_data.sizes['time']
    else:
        n_times = 1
    
    # Initialize results
    daily_means = []
    daily_stds = []
    daily_counts = []
    daily_timestamps = []
    
    # Process based on variable type
    if variable_type == 'depth_averaged':
        # Get depth levels
        depth_levels = variable_config['depth_levels']
        
        # Find depth dimension
        depth_dim = None
        for dim in var_data.dims:
            if dim not in ['time', 'xi_rho', 'eta_rho'] and dim in variable_ds.coords:
                depth_dim = dim
                break
        
        # Get depth indices
        depth_indices = []
        if depth_dim and 'depth' in variable_ds:
            # Get all depth values
            depth_values = variable_ds['depth'].values
            
            # Find indices of requested depth levels
            for target_depth in depth_levels:
                # Find nearest depth
                closest_idx = np.abs(depth_values - target_depth).argmin()
                depth_indices.append(closest_idx)
        else:
            # Use first few depth levels
            if hasattr(var_data, 'dims') and 'depth' in var_data.dims:
                depth_indices = list(range(min(len(depth_levels), var_data.sizes['depth'])))
            else:
                depth_indices = list(range(min(len(depth_levels), len(depth_levels))))
        
        # Process each time step
        for t in range(n_times):
            # Calculate the average across depth layers first
            depth_slices = []
            for d_idx in depth_indices:
                try:
                    if n_times > 1:
                        slice_data = var_data.isel(time=t, depth=d_idx)
                    else:
                        slice_data = var_data.isel(depth=d_idx)
                    depth_slices.append(slice_data)
                except Exception as e:
                    print(f"Warning: Could not extract slice at time={t}, depth={d_idx}: {e}")
                    continue
            
            # Stack depth slices and calculate mean across depth
            if depth_slices:
                try:
                    stacked = xr.concat(depth_slices, dim='z_levels')
                    depth_mean = stacked.mean(dim='z_levels', skipna=True)
                    
                    # Apply the mask
                    if depth_mean.shape != mask.shape:
                        print(f"Warning: Mask shape {mask.shape} doesn't match data shape {depth_mean.shape}")
                        print(f"Calculating statistics only on the valid overlapping region")
                        
                        # Find the minimum dimensions that can be used
                        min_rows = min(mask.shape[0], depth_mean.shape[0])
                        min_cols = min(mask.shape[1], depth_mean.shape[1])
                        
                        # Truncate both mask and data to the common region
                        usable_mask = mask[:min_rows, :min_cols]
                        usable_data = depth_mean.values[:min_rows, :min_cols]
                        
                        # Apply truncated mask to truncated data
                        masked_values = usable_data.copy()
                        masked_values[~usable_mask] = np.nan
                    else:
                        # Apply the mask directly if shapes match
                        masked_values = depth_mean.values.copy()
                        masked_values[~mask] = np.nan
                    
                    # Calculate statistics if there are valid points
                    if len(masked_values) > 0:
                        daily_means.append(np.nanmean(masked_values))
                        daily_stds.append(np.nanstd(masked_values))
                        daily_counts.append(np.sum(~np.isnan(masked_values)))
                    else:
                        daily_means.append(np.nan)
                        daily_stds.append(np.nan)
                        daily_counts.append(0)
                    
                    # Get timestamp if available
                    if n_times > 1 and hasattr(var_data, 'time') and hasattr(var_data.time, 'values'):
                        daily_timestamps.append(var_data.time.values[t])
                    else:
                        daily_timestamps.append(t)
                except Exception as e:
                    print(f"Error processing depth slices: {e}")
                    daily_means.append(np.nan)
                    daily_stds.append(np.nan)
                    daily_counts.append(0)
                    daily_timestamps.append(t)
    
    elif variable_type == 'single_depth':
        # Get target depth
        target_depth = variable_config['target_depth']
        
        # Find the closest depth value
        if 'depth' in variable_ds:
            depth_values = variable_ds['depth'].values
            depth_idx = np.abs(depth_values - target_depth).argmin()
            actual_depth = depth_values[depth_idx]
            print(f"Using depth value {actual_depth}m (closest to target {target_depth}m)")
        else:
            # Default to first depth level
            depth_idx = 0
            print(f"Warning: No depth coordinate found, using depth index {depth_idx}")
        
        # Process each time step
        for t in range(n_times):
            try:
                # Extract data for the target depth
                if n_times > 1:
                    depth_slice = var_data.isel(time=t, depth=depth_idx)
                else:
                    depth_slice = var_data.isel(depth=depth_idx)
                
                # Check if mask and data dimensions match
                if depth_slice.shape != mask.shape:
                    print(f"Warning: Mask shape {mask.shape} doesn't match data shape {depth_slice.shape} at time {t}, depth {depth_idx}")
                    print(f"Calculating statistics only on the valid overlapping region")
                    
                    # Find the minimum dimensions that can be used
                    min_rows = min(mask.shape[0], depth_slice.shape[0])
                    min_cols = min(mask.shape[1], depth_slice.shape[1])
                    
                    # Truncate both mask and data to the common region
                    usable_mask = mask[:min_rows, :min_cols]
                    usable_data = depth_slice.values[:min_rows, :min_cols]
                    
                    # Apply truncated mask to truncated data
                    masked_values = usable_data.copy()
                    masked_values[~usable_mask] = np.nan
                else:
                    # Apply the mask directly if shapes match
                    masked_values = depth_slice.values.copy()
                    masked_values[~mask] = np.nan
                
                # Calculate statistics on masked values
                valid_values = masked_values[~np.isnan(masked_values)]
                
                if len(valid_values) > 0:
                    if variable_name in UNIT_CONVERSION:
                        # For POC_FLUX_IN, take absolute value of EACH grid point FIRST, then calculate mean
                        if variable_name == 'POC_FLUX_IN':
                            daily_means.append(np.mean(np.abs(valid_values)) * UNIT_CONVERSION[variable_name])
                        else:
                            daily_means.append(np.mean(valid_values) * UNIT_CONVERSION[variable_name])
                    else:
                        daily_means.append(np.mean(valid_values))
                    daily_stds.append(np.nanstd(masked_values))
                    daily_counts.append(np.sum(~np.isnan(valid_values)))
                else:
                    daily_means.append(np.nan)
                    daily_stds.append(np.nan)
                    daily_counts.append(0)
                
                # Get timestamp if available
                if n_times > 1 and hasattr(var_data, 'time') and hasattr(var_data.time, 'values'):
                    daily_timestamps.append(var_data.time.values[t])
                else:
                    daily_timestamps.append(t)
            except Exception as e:
                print(f"Error processing time step {t}: {e}")
                daily_means.append(np.nan)
                daily_stds.append(np.nan)
                daily_counts.append(0)
    
    # Create results dictionary
    stats = {
        'daily_means': np.array(daily_means),
        'daily_stds': np.array(daily_stds), 
        'daily_counts': np.array(daily_counts),
        'timestamps': np.array(daily_timestamps)
    }
    
    return stats

def load_all_climatology_data(month, years_range=ALL_YEARS):
    """Load data for the same month across specified years."""
    # Check if month is in analysis range
    if month not in ANALYSIS_MONTHS:
        print(f"Month {month} is outside the April-November analysis window")
        return {}
        
    # Format month with leading zeros
    month_str = f"{month:03d}"
    
    # Dictionary to hold datasets by year
    datasets = {}
    
    print(f"\nLoading climatology data for month {month} across years {min(years_range)}-{max(years_range)}...")
    
    # Load data for each year
    for year in years_range:
        file_pattern = os.path.join(DATA_DIR, f"z_avg_{year}_{month_str}_37zlevs_full_1x1meanpool_downsampling.nc")
        matching_files = glob.glob(file_pattern)
        
        if matching_files:
            try:
                ds = xr.open_dataset(matching_files[0])
                # Check if at least one of our target variables is present
                if any(var_config['name'] in ds for var_config in VARIABLES):
                    datasets[year] = ds
                    print(f"  ✓ Loaded data for {year}-{month:02d}")
                else:
                    print(f"  × Required variables not found in dataset for {year}")
            except Exception as e:
                print(f"  × Error loading data for {year}: {e}")
    
    print(f"Found data for {len(datasets)} years for climatology calculation")
    return datasets

def calculate_climatology_stats(clim_datasets, variable_config, mask):
    """Calculate climatological statistics using the provided mask."""
    # Initialize arrays to hold daily statistics across years
    all_datasets_means = []  # Will be list of lists [year][day]
    
    variable_name = variable_config['name']
    print(f"\nCalculating climatology statistics for {variable_name}...")
    
    # Process each year's dataset
    valid_years = 0
    for year, ds in clim_datasets.items():
        # Skip datasets that don't have this variable
        if variable_name not in ds:
            print(f"  × Variable {variable_name} not found in dataset for {year}")
            continue
        
        valid_years += 1
        # Calculate stats for this year using the provided mask
        year_stats = calculate_masked_variable_stats(ds, variable_config, mask)
        
        if year_stats is None:
            continue
            
        # Add to collection
        all_datasets_means.append(year_stats['daily_means'])
    
    if valid_years == 0:
        print(f"ERROR: No valid years found with {variable_name} data for climatology")
        return None
        
    # Find longest month length (in case of different lengths)
    max_days = max([len(means) for means in all_datasets_means])
    
    # Initialize climatology arrays
    clim_means = np.zeros(max_days)
    clim_stds = np.zeros(max_days)
    day_counts = np.zeros(max_days, dtype=int)
    
    # Calculate climatology for each day
    for day_idx in range(max_days):
        # Collect all valid means for this day across years
        day_means = []
        
        for year_means in all_datasets_means:
            if day_idx < len(year_means) and not np.isnan(year_means[day_idx]):
                day_means.append(year_means[day_idx])
        
        # If we have data for this day, calculate statistics
        if day_means:
            clim_means[day_idx] = np.mean(day_means)
            clim_stds[day_idx] = np.std(day_means)
            day_counts[day_idx] = len(day_means)
        else:
            clim_means[day_idx] = np.nan
            clim_stds[day_idx] = np.nan
            day_counts[day_idx] = 0
    
    # Create climatology stats dictionary
    clim_stats = {
        'daily_means': clim_means,
        'daily_stds': clim_stds,
        'day_counts': day_counts
    }
    
    print(f"Calculated climatology for {variable_name} with averaging {np.mean(day_counts):.1f} years per day")
    
    return clim_stats

# %%
#--------- PERIOD PROCESSING FUNCTIONS ---------#
def process_domain_month(year, month, variable_config):
    """Process a single month using full domain mask (for pre/post-blob periods)."""
    # Check if month is in analysis range
    if month not in ANALYSIS_MONTHS:
        return None, None
        
    variable_name = variable_config['name']
    
    # Load variable data
    print(f"\nLoading {variable_name} data for {year}-{month:02d}...")
    variable_ds = load_variable_data(year, month)
    if variable_ds is None:
        print(f"ERROR: Could not load data for {year}-{month:02d}")
        return None, None
    
    # Check if variable exists in the dataset
    if variable_name not in variable_ds:
        print(f"WARNING: Variable {variable_name} not found in {year}-{month:02d} dataset. Skipping.")
        return None, None
    
    # Create domain mask for the full study region
    print(f"\nCreating mask for full study domain...")
    domain_mask = create_domain_mask(variable_ds)
    if domain_mask is None:
        print(f"ERROR: Could not create domain mask for {year}-{month:02d}")
        return None, None
    
    # Calculate statistics for the variable over the entire domain
    print(f"\nCalculating {variable_name} statistics over entire domain...")
    stats = calculate_masked_variable_stats(variable_ds, variable_config, domain_mask)
    if stats is None:
        print(f"ERROR: Could not calculate statistics for {variable_name} in {year}-{month:02d}")
        return None, None
    
    # Load climatology data (all years 2011-2021)
    clim_datasets = load_all_climatology_data(month)
    if not clim_datasets:
        print(f"ERROR: Could not load sufficient data for climatology for {month:02d}")
        return stats, None
    
    # Calculate climatology statistics using the same domain mask
    print(f"\nCalculating climatology for whole domain...")
    clim_stats = calculate_climatology_stats(clim_datasets, variable_config, domain_mask)
    if clim_stats is None:
        print(f"ERROR: Could not calculate climatology for {variable_name} in {year}-{month:02d}")
        return stats, None
    
    # Generate dates for this month
    dates = []
    for i in range(len(stats['daily_means'])):
        day = i + 1
        try:
            dates.append(pd.Timestamp(f"{year}-{month:02d}-{day:02d}"))
        except ValueError:
            # Handle cases where the day is invalid
            if i > 0:
                dates.append(dates[-1] + pd.Timedelta(days=1))
            else:
                dates.append(pd.Timestamp(f"{year}-{month:02d}-01"))
    
    # Add dates to stats
    stats['dates'] = dates
    
    return stats, clim_stats

def process_blob_month(bool_ds, year, month, variable_config):
    """
    Process a single month using MHW contour masks that are depth-resolved.
    For depth-averaged variables, separate masks are created and applied for each depth.
    """
    # Check if month is in analysis range
    if month not in ANALYSIS_MONTHS:
        return None, None
        
    global MHW_THRESHOLD_MIN  # Proper global declaration
    variable_name = variable_config['name']
    variable_type = variable_config['type']
    
    # Load variable data
    print(f"\nLoading {variable_name} data for {year}-{month:02d}...")
    variable_ds = load_variable_data(year, month)
    if variable_ds is None:
        print(f"ERROR: Could not load data for {year}-{month:02d}")
        return None, None
    
    # Check if variable exists in the dataset
    if variable_name not in variable_ds:
        print(f"WARNING: {variable_name} not found in {year}-{month:02d} dataset. Skipping.")
        return None, None
    
    min_threshold = MHW_THRESHOLD_MIN
    if isinstance(MHW_THRESHOLD_MIN, list):
        min_threshold = MHW_THRESHOLD_MIN[0]
        
    # Handle depth-specific processing
    if variable_type == 'single_depth':
        # For single-depth variables, create a single mask at the target depth
        target_depth = variable_config['target_depth']
        
        # Find the closest depth value in both the variable dataset and boolean dataset
        if 'depth' in variable_ds and 'depth' in bool_ds:
            var_depths = variable_ds['depth'].values
            bool_depths = bool_ds['depth'].values
            
            # Find closest depth index in the variable dataset
            var_depth_idx = np.abs(var_depths - target_depth).argmin()
            actual_var_depth = var_depths[var_depth_idx]
            
            # Find closest depth index in the boolean dataset
            bool_depth_idx = np.abs(bool_depths - target_depth).argmin()
            actual_bool_depth = bool_depths[bool_depth_idx]
            
            print(f"Using depth {actual_var_depth}m (var dataset) and {actual_bool_depth}m (bool dataset)")
        else:
            # Default to first depth level if depth coordinate not found
            var_depth_idx = 0
            bool_depth_idx = 0
            print(f"Warning: Using default depth index 0 (depth coordinates not found)")
        
        # Get extreme event frequency for the specific depth
        monthly_freq_result = get_monthly_extreme_data(bool_ds, year, month, depth_idx=bool_depth_idx)
        if monthly_freq_result is None:
            print(f"ERROR: Could not calculate MHW frequency for {year}-{month:02d} at depth index {bool_depth_idx}")
            return None, None
        
        monthly_freq, time_indices = monthly_freq_result
        
        # Create mask for this depth
        print(f"Creating mask for MHW frequency between {min_threshold} and {MHW_THRESHOLD_MAX} at depth index {bool_depth_idx}...")
        mask = create_mhw_contour_mask(monthly_freq, min_threshold, MHW_THRESHOLD_MAX)
        
        # Calculate statistics for the variable at the target depth with the corresponding mask
        print(f"Calculating {variable_name} statistics at depth {actual_var_depth}m with MHW mask...")
        
        # Override normal calculation with single depth mask
        single_depth_config = variable_config.copy()
        single_depth_config['target_depth_idx'] = var_depth_idx  # Add depth index for use in calculation
        
        stats = calculate_single_depth_stats(variable_ds, single_depth_config, mask)
        
    else:  # For depth-averaged variables
        # Get depth levels
        depth_levels = variable_config['depth_levels']
        
        # We'll store depth-specific data and then average
        depth_stats_list = []
        
        # Process each depth level
        for depth_level in depth_levels:
            # Find closest depth in both datasets
            if 'depth' in variable_ds and 'depth' in bool_ds:
                var_depths = variable_ds['depth'].values
                bool_depths = bool_ds['depth'].values
                
                # Find closest depth indices
                var_depth_idx = np.abs(var_depths - depth_level).argmin()
                actual_var_depth = var_depths[var_depth_idx]
                
                bool_depth_idx = np.abs(bool_depths - depth_level).argmin()
                actual_bool_depth = bool_depths[bool_depth_idx]
                
                print(f"Processing depth {depth_level}m: Using {actual_var_depth}m (var) and {actual_bool_depth}m (bool)")
            else:
                # Default to index mapping if depth coords not found
                # Assuming depth indices match between datasets
                var_depth_idx = depth_levels.index(depth_level) if depth_level in depth_levels else 0
                bool_depth_idx = var_depth_idx
                print(f"Warning: Using depth indices directly: {var_depth_idx} (no depth coordinates found)")
            
            # Get extreme event frequency for this depth
            monthly_freq_result = get_monthly_extreme_data(bool_ds, year, month, depth_idx=bool_depth_idx)
            if monthly_freq_result is None:
                print(f"Error: Could not get MHW data for depth {depth_level}m. Skipping depth.")
                continue
                
            monthly_freq, time_indices = monthly_freq_result
            
            # Create mask for this specific depth
            print(f"Creating mask for MHW frequency between {min_threshold} and {MHW_THRESHOLD_MAX} at depth {depth_level}m...")
            depth_mask = create_mhw_contour_mask(monthly_freq, min_threshold, MHW_THRESHOLD_MAX)
            
            # Calculate stats for this depth only
            print(f"Calculating {variable_name} statistics at depth {actual_var_depth}m with corresponding MHW mask...")
            
            # Create a temporary single depth config
            depth_config = {
                'name': variable_name,
                'type': 'single_depth', 
                'target_depth': depth_level,
                'target_depth_idx': var_depth_idx  # Pass the index directly
            }
            
            # Get stats for this single depth
            depth_stats = calculate_single_depth_stats(variable_ds, depth_config, depth_mask)
            
            if depth_stats is not None:
                depth_stats_list.append(depth_stats)
        
        # Combine depth-specific statistics into a single set
        if not depth_stats_list:
            print(f"Error: No valid depth statistics for {variable_name}. Skipping.")
            return None, None
            
        # Average the means across depths
        stats = average_depth_stats(depth_stats_list)
        print(f"Averaged statistics across {len(depth_stats_list)} depth levels")
    
    # Load climatology data
    clim_datasets = load_all_climatology_data(month)
    if not clim_datasets:
        print(f"ERROR: Could not load sufficient data for climatology for {month:02d}")
        return stats, None
    
    # Calculate climatology statistics using the SAME approach as for the actual data
    if variable_type == 'single_depth':
        # Single depth climatology using the same mask
        clim_stats = calculate_single_depth_climatology(clim_datasets, single_depth_config, mask)
    else:
        # Depth-averaged climatology with depth-specific masks
        depth_clim_list = []
        
        for depth_level in depth_levels:
            # Find depth indices (similar to above)
            if 'depth' in variable_ds and 'depth' in bool_ds:
                var_depths = variable_ds['depth'].values
                bool_depths = bool_ds['depth'].values
                var_depth_idx = np.abs(var_depths - depth_level).argmin()
                bool_depth_idx = np.abs(bool_depths - depth_level).argmin()
            else:
                var_depth_idx = depth_levels.index(depth_level) if depth_level in depth_levels else 0
                bool_depth_idx = var_depth_idx
            
            # Get MHW mask for this depth
            monthly_freq_result = get_monthly_extreme_data(bool_ds, year, month, depth_idx=bool_depth_idx)
            if monthly_freq_result is None:
                print(f"Error: No MHW data for depth {depth_level}m for climatology. Skipping.")
                continue
                
            monthly_freq, _ = monthly_freq_result
            depth_mask = create_mhw_contour_mask(monthly_freq, min_threshold, MHW_THRESHOLD_MAX)
            
            # Create depth config
            depth_config = {
                'name': variable_name,
                'type': 'single_depth',
                'target_depth': depth_level,
                'target_depth_idx': var_depth_idx
            }
            
            # Calculate climatology for this depth
            depth_clim = calculate_single_depth_climatology(clim_datasets, depth_config, depth_mask)
            
            if depth_clim is not None:
                depth_clim_list.append(depth_clim)
        
        # Average climatologies across depths
        if not depth_clim_list:
            print("Error: No valid depth climatologies. Skipping.")
            return stats, None
            
        clim_stats = average_depth_stats(depth_clim_list)
    
    # Generate dates
    dates = []
    for i in range(len(stats['daily_means'])):
        day = i + 1
        try:
            dates.append(pd.Timestamp(f"{year}-{month:02d}-{day:02d}"))
        except ValueError:
            if i > 0:
                dates.append(dates[-1] + pd.Timedelta(days=1))
            else:
                dates.append(pd.Timestamp(f"{year}-{month:02d}-01"))
    
    # Add dates to stats
    stats['dates'] = dates
    
    return stats, clim_stats

def calculate_single_depth_stats(ds, depth_config, mask):
    """
    Calculate statistics for a variable at a single depth with the provided mask.
    """
    variable_name = depth_config['name']
    depth_idx = depth_config.get('target_depth_idx', 0)
    
    # Get variable data
    if variable_name not in ds:
        print(f"ERROR: Variable {variable_name} not found in dataset")
        return None
        
    var_data = ds[variable_name]
    
    # Get time dimension size
    n_times = var_data.sizes['time'] if 'time' in var_data.dims else 1
    
    # Initialize results
    daily_means = []
    daily_stds = []
    daily_counts = []
    daily_timestamps = []  
    
    # Process each timestep
    for t in range(n_times):
        try:
            # Extract data slice for the target depth and time
            if n_times > 1:
                depth_slice = var_data.isel(time=t, depth=depth_idx)
            else:
                depth_slice = var_data.isel(depth=depth_idx)
            
            # Check if mask and data dimensions match
            if depth_slice.shape != mask.shape:
                print(f"Warning: Mask shape {mask.shape} doesn't match data shape {depth_slice.shape} at time {t}, depth {depth_idx}")
                print(f"Calculating statistics only on the valid overlapping region")
                
                # Find the minimum dimensions that can be used
                min_rows = min(mask.shape[0], depth_slice.shape[0])
                min_cols = min(mask.shape[1], depth_slice.shape[1])
                
                # Truncate both mask and data to the common region
                usable_mask = mask[:min_rows, :min_cols]
                usable_data = depth_slice.values[:min_rows, :min_cols]
                
                # Apply truncated mask to truncated data
                masked_values = usable_data.copy()
                masked_values[~usable_mask] = np.nan
            else:
                # Apply the mask directly if shapes match
                masked_values = depth_slice.values.copy()
                masked_values[~mask] = np.nan
            
            # Calculate statistics on masked values
            valid_values = masked_values[~np.isnan(masked_values)]
            
            if len(valid_values) > 0:
                if variable_name in UNIT_CONVERSION:
                    # For POC_FLUX_IN, take absolute value of EACH grid point FIRST, then calculate mean
                    if variable_name == 'POC_FLUX_IN':
                        daily_means.append(np.mean(np.abs(valid_values)) * UNIT_CONVERSION[variable_name])
                    else:
                        daily_means.append(np.mean(valid_values) * UNIT_CONVERSION[variable_name])
                else:
                    daily_means.append(np.mean(valid_values))
                daily_stds.append(np.nanstd(masked_values))
                daily_counts.append(np.sum(~np.isnan(valid_values)))
            else:
                daily_means.append(np.nan)
                daily_stds.append(np.nan)
                daily_counts.append(0)
            
            # Get timestamp if available
            if n_times > 1 and hasattr(var_data, 'time') and hasattr(var_data.time, 'values'):
                daily_timestamps.append(var_data.time.values[t])
            else:
                daily_timestamps.append(t)
        except Exception as e:
            print(f"Error processing time step {t}: {e}")
            daily_means.append(np.nan)
            daily_stds.append(np.nan)
            daily_counts.append(0)
    
    # Create results dictionary
    stats = {
        'daily_means': np.array(daily_means),
        'daily_stds': np.array(daily_stds), 
        'daily_counts': np.array(daily_counts),
        'timestamps': np.array(daily_timestamps)
    }
    
    return stats

def calculate_single_depth_climatology(clim_datasets, depth_config, mask):
    """
    Calculate climatology for a specific depth using the provided mask.
    """
    variable_name = depth_config['name']
    depth_idx = depth_config.get('target_depth_idx', 0)
    
    print(f"Calculating climatology for {variable_name} at depth index {depth_idx}")
    
    # Initialize arrays to hold daily values across years
    all_daily_means = []  # List of lists [year][day]
    all_daily_counts = []  # To track valid points
    
    # Process each year's dataset
    valid_years = 0
    for year, ds in clim_datasets.items():
        if variable_name not in ds:
            continue
            
        valid_years += 1
        var_data = ds[variable_name]
        
        # Get time dimension size
        n_times = var_data.sizes['time'] if 'time' in var_data.dims else 1
        
        # Process each timestep
        daily_means = []
        daily_counts = []
        
        for t in range(n_times):
            try:
                # Extract data slice for this depth and time
                if n_times > 1:
                    depth_slice = var_data.isel(time=t, depth=depth_idx)
                else:
                    depth_slice = var_data.isel(depth=depth_idx)
                
                # Check if mask and data dimensions match
                if depth_slice.shape != mask.shape:
                    print(f"Warning: Mask shape {mask.shape} doesn't match data shape {depth_slice.shape} at year {year}, time {t}")
                    print(f"Calculating statistics only on the valid overlapping region")
                    
                    # Find the minimum dimensions that can be used
                    min_rows = min(mask.shape[0], depth_slice.shape[0])
                    min_cols = min(mask.shape[1], depth_slice.shape[1])
                    
                    # Truncate both mask and data to the common region
                    usable_mask = mask[:min_rows, :min_cols]
                    usable_data = depth_slice.values[:min_rows, :min_cols]
                    
                    # Apply truncated mask to truncated data
                    masked_values = usable_data.copy()
                    masked_values[~usable_mask] = np.nan
                else:
                    # Apply the mask directly if shapes match
                    masked_values = depth_slice.values.copy()
                    masked_values[~mask] = np.nan

                # Calculate statistics if there are valid points
                valid_values = masked_values[~np.isnan(masked_values)]
                if len(valid_values) > 0:
                    if variable_name in UNIT_CONVERSION:
                        # For POC_FLUX_IN, take absolute value
                        if variable_name == 'POC_FLUX_IN':
                            daily_means.append(np.abs(np.mean(valid_values)) * UNIT_CONVERSION[variable_name])
                        else:
                            daily_means.append(np.mean(valid_values) * UNIT_CONVERSION[variable_name])
                    else:
                        daily_means.append(np.mean(valid_values))
                    
                    # Add the count
                    daily_counts.append(len(valid_values))
                else:
                    daily_means.append(np.nan)
                    daily_counts.append(0)
                    
            except Exception as e:
                print(f"Error processing year {year}, time {t}: {e}")
                daily_means.append(np.nan)
                daily_counts.append(0)
        
        all_daily_means.append(daily_means)
        all_daily_counts.append(daily_counts)
        
    max_days = max([len(means) for means in all_daily_means])
    # Initialize climatology arrays
    clim_means = np.zeros(max_days)
    clim_stds = np.zeros(max_days)
    day_counts = np.zeros(max_days, dtype=int)
    
    # Calculate climatology for each day
    for day_idx in range(max_days):
        # Collect all valid means for this day across years
        day_means = []
        day_weights = []
        
        for y in range(len(all_daily_means)):
            # Make sure both means and counts have this index
            if (day_idx < len(all_daily_means[y]) and 
                day_idx < len(all_daily_counts[y]) and 
                not np.isnan(all_daily_means[y][day_idx])):
                
                day_means.append(all_daily_means[y][day_idx])
                # Use count as weight
                day_weights.append(all_daily_counts[y][day_idx] if all_daily_counts[y][day_idx] > 0 else 1)
        
        # If we have data, calculate weighted statistics
        if day_means:
            weights = np.array(day_weights) / sum(day_weights)
            clim_means[day_idx] = np.sum(np.array(day_means) * weights)
            if len(day_means) > 1:
                clim_stds[day_idx] = np.sqrt(np.sum(weights * np.square(np.array(day_means) - clim_means[day_idx])))
            else:
                clim_stds[day_idx] = 0
            day_counts[day_idx] = len(day_means)
        else:
            clim_means[day_idx] = np.nan
            clim_stds[day_idx] = np.nan
            day_counts[day_idx] = 0
    
    # Create climatology stats dictionary
    clim_stats = {
        'daily_means': clim_means,
        'daily_stds': clim_stds,
        'day_counts': day_counts
    }
    
    return clim_stats

# Add this to main() to use the new depth-resolved implementation
def main():
    """Main execution function."""
    print("-" * 80)
    print(f"COMPLETE TIMESERIES ANALYSIS (2011-2021) WITH DEPTH-RESOLVED MHW MASKS")
    print(f"Pre-Blob (2011-2013): Full Domain")
    print(f"Blob (2014-2016): Marine Heatwave Areas (depth-specific masks)")
    print(f"Post-Blob (2017-2021): Full Domain")
    print(f"MONTHS: April (003) through November (011) only")
    print("-" * 80)
    
    # Remaining code stays the same
    # ...

def process_complete_timeseries(variable_config, bool_ds):
    """Process complete timeseries for a variable with multiple thresholds."""
    global MHW_THRESHOLD_MIN
    
    variable_name = variable_config['name']
    variable_title = variable_config['title']
    variable_type = variable_config['type']
    
    # Define units based on variable
    if variable_name == 'temp':
        units = "°C"
    elif variable_name == 'TOT_PROD':
        units = "mol/m³/y"
    elif variable_name == 'POC_FLUX_IN':
        units = "mol/m²/y"
    else:
        units = ""
    
    print(f"\nProcessing {variable_title} timeseries (Apr-Nov only)...")
    
    # Keep this global declaration
    global MHW_THRESHOLD_MIN
    original_threshold_array = MHW_THRESHOLD_MIN.copy() if isinstance(MHW_THRESHOLD_MIN, list) else MHW_THRESHOLD_MIN
    
    # Calculate pre-blob and post-blob data once
    print("Calculating pre-blob and post-blob data once (will be reused)...")
    
    # Data containers for pre-blob and post-blob periods
    pre_blob_data = {'dates': [], 'means': [], 'stds': [], 
                    'clim_means': [], 'clim_stds': [], 'anomalies': []}
    post_blob_data = {'dates': [], 'means': [], 'stds': [], 
                     'clim_means': [], 'clim_stds': [], 'anomalies': []}
    
    # Process Pre-Blob period (2011-2013) - WHOLE DOMAIN (only once)
    print(f"Processing pre-blob period (2011-2013)...")
    for year in PRE_BLOB_YEARS:
        for month in ANALYSIS_MONTHS:
            stats, clim_stats = process_domain_month(year, month, variable_config)
            if stats is not None and clim_stats is not None:
                # Calculate anomalies
                anomalies = stats['daily_means'] - clim_stats['daily_means'][:len(stats['daily_means'])]
                
                # Add to pre-blob data
                pre_blob_data['dates'].extend(stats['dates'])
                pre_blob_data['means'].extend(stats['daily_means'])
                pre_blob_data['stds'].extend(stats['daily_stds'])
                pre_blob_data['clim_means'].extend(clim_stats['daily_means'][:len(stats['daily_means'])])
                pre_blob_data['clim_stds'].extend(clim_stats['daily_stds'][:len(stats['daily_means'])])
                pre_blob_data['anomalies'].extend(anomalies)
    
    # Process Post-Blob period (2017-2021) - WHOLE DOMAIN (only once)
    print(f"Processing post-blob period (2017-2021)...")
    for year in POST_BLOB_YEARS:
        for month in ANALYSIS_MONTHS:
            stats, clim_stats = process_domain_month(year, month, variable_config)
            if stats is not None and clim_stats is not None:
                # Calculate anomalies
                anomalies = stats['daily_means'] - clim_stats['daily_means'][:len(stats['daily_means'])]
                
                # Add to post-blob data
                post_blob_data['dates'].extend(stats['dates'])
                post_blob_data['means'].extend(stats['daily_means'])
                post_blob_data['stds'].extend(stats['daily_stds'])
                post_blob_data['clim_means'].extend(clim_stats['daily_means'][:len(stats['daily_means'])])
                post_blob_data['clim_stds'].extend(clim_stats['daily_stds'][:len(stats['daily_means'])])
                post_blob_data['anomalies'].extend(anomalies)
    
    # Calculate blob period data using WHOLE DOMAIN approach if requested
    whole_region_blob_data = None
    if INCLUDE_WHOLE_REGION:
        print(f"Processing blob period (2014-2016) using WHOLE REGION approach for comparison...")
        whole_region_blob_data = {'dates': [], 'means': [], 'stds': [], 
                                  'clim_means': [], 'clim_stds': [], 'anomalies': []}
        
        for year in BLOB_YEARS:
            for month in ANALYSIS_MONTHS:
                stats, clim_stats = process_domain_month(year, month, variable_config)
                if stats is not None and clim_stats is not None:
                    # Calculate anomalies
                    anomalies = stats['daily_means'] - clim_stats['daily_means'][:len(stats['daily_means'])]
                    
                    # Add to whole region blob data
                    whole_region_blob_data['dates'].extend(stats['dates'])
                    whole_region_blob_data['means'].extend(stats['daily_means'])
                    whole_region_blob_data['stds'].extend(stats['daily_stds'])
                    whole_region_blob_data['clim_means'].extend(clim_stats['daily_means'][:len(stats['daily_means'])])
                    whole_region_blob_data['clim_stds'].extend(clim_stats['daily_stds'][:len(stats['daily_means'])])
                    whole_region_blob_data['anomalies'].extend(anomalies)
    
    # Now process each threshold, reusing pre-blob and post-blob data
    for i, min_threshold in enumerate(original_threshold_array if isinstance(original_threshold_array, list) else [original_threshold_array]):
        print(f"  Processing data for MHW threshold {min_threshold*100:.0f}%")
        
        # Temporarily set the threshold to the current value (for the blob period processing)
        MHW_THRESHOLD_MIN = min_threshold
        
        # Process Blob period (2014-2016) - needs to be recalculated for each threshold
        blob_data = {'dates': [], 'means': [], 'stds': [], 
                    'clim_means': [], 'clim_stds': [], 'anomalies': []}
        
        print(f"  Processing blob period (2014-2016) with threshold {min_threshold*100:.0f}%...")
        if bool_ds is not None:
            for year in BLOB_YEARS:
                for month in ANALYSIS_MONTHS:
                    stats, clim_stats = process_blob_month(bool_ds, year, month, variable_config)
                    if stats is not None and clim_stats is not None:
                        # Calculate anomalies
                        anomalies = stats['daily_means'] - clim_stats['daily_means'][:len(stats['daily_means'])]
                        
                        # Add to blob data
                        blob_data['dates'].extend(stats['dates'])
                        blob_data['means'].extend(stats['daily_means'])
                        blob_data['stds'].extend(stats['daily_stds'])
                        blob_data['clim_means'].extend(clim_stats['daily_means'][:len(stats['daily_means'])])
                        blob_data['clim_stds'].extend(clim_stats['daily_stds'][:len(stats['daily_means'])])
                        blob_data['anomalies'].extend(anomalies)
        
        # Combine all data for this threshold
        all_data = {
            'date': [],
            'value': [],
            'std': [],
            'climatology': [],
            'clim_std': [],
            'anomaly': [],
            'period': [],
            'method': []  # Add method column to distinguish between MHW mask and whole region
        }
        
        # Add pre-blob data
        if pre_blob_data['dates']:
            for i in range(len(pre_blob_data['dates'])):
                all_data['date'].append(pre_blob_data['dates'][i])
                all_data['value'].append(pre_blob_data['means'][i])
                all_data['std'].append(pre_blob_data['stds'][i])
                all_data['climatology'].append(pre_blob_data['clim_means'][i])
                all_data['clim_std'].append(pre_blob_data['clim_stds'][i])
                all_data['anomaly'].append(pre_blob_data['anomalies'][i])
                all_data['period'].append('pre_blob')
                all_data['method'].append('whole_region')
        
        # Add blob data
        if blob_data['dates']:
            for i in range(len(blob_data['dates'])):
                all_data['date'].append(blob_data['dates'][i])
                all_data['value'].append(blob_data['means'][i])
                all_data['std'].append(blob_data['stds'][i])
                all_data['climatology'].append(blob_data['clim_means'][i])
                all_data['clim_std'].append(blob_data['clim_stds'][i])
                all_data['anomaly'].append(blob_data['anomalies'][i])
                all_data['period'].append('blob')
                all_data['method'].append('mhw_mask')
        
        # Add whole region blob data if available
        if INCLUDE_WHOLE_REGION and whole_region_blob_data and whole_region_blob_data['dates']:
            for i in range(len(whole_region_blob_data['dates'])):
                all_data['date'].append(whole_region_blob_data['dates'][i])
                all_data['value'].append(whole_region_blob_data['means'][i])
                all_data['std'].append(whole_region_blob_data['stds'][i])
                all_data['climatology'].append(whole_region_blob_data['clim_means'][i])
                all_data['clim_std'].append(whole_region_blob_data['clim_stds'][i])
                all_data['anomaly'].append(whole_region_blob_data['anomalies'][i])
                all_data['period'].append('blob')
                all_data['method'].append('whole_region')
        
        # Add post-blob data
        if post_blob_data['dates']:
            for i in range(len(post_blob_data['dates'])):
                all_data['date'].append(post_blob_data['dates'][i])
                all_data['value'].append(post_blob_data['means'][i])
                all_data['std'].append(post_blob_data['stds'][i])
                all_data['climatology'].append(post_blob_data['clim_means'][i])
                all_data['clim_std'].append(post_blob_data['clim_stds'][i])
                all_data['anomaly'].append(post_blob_data['anomalies'][i])
                all_data['period'].append('post_blob')
                all_data['method'].append('whole_region')
        
        # Create dataframe and sort by date
        df = pd.DataFrame(all_data)
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
        
        # Get date range for plotting
        dates = df.index.to_list()
        
        # Create plotting figure - add comparison plot if whole region is included
        if INCLUDE_WHOLE_REGION:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16), sharex=True, 
                                           gridspec_kw={'height_ratios': [1, 1, 1]})
            
            # Extract data for plotting (MHW mask approach)
            mhw_mask_df = df[df['method'] == 'mhw_mask']
            mhw_data_values = mhw_mask_df['value'].values
            mhw_clim_values = mhw_mask_df['climatology'].values
            mhw_anomaly_values = mhw_mask_df['anomaly'].values
            mhw_dates = mhw_mask_df.index.to_list()
            
            # Extract data for plotting (whole region approach)
            whole_region_df = df[df['method'] == 'whole_region']
            whole_region_data_values = whole_region_df['value'].values
            whole_region_clim_values = whole_region_df['climatology'].values
            whole_region_anomaly_values = whole_region_df['anomaly'].values
            whole_region_dates = whole_region_df.index.to_list()
            
            # Top plot: Data values with climatology (MHW mask)
            ax1.plot(mhw_dates, mhw_data_values, 'b-', linewidth=2, label='Data (MHW mask)')
            ax1.plot(mhw_dates, mhw_clim_values, 'r--', linewidth=1.5, label='Climatology (MHW mask)')
            
            # Mark blob period with vertical spans
            blob_start = pd.Timestamp(f"{BLOB_YEARS[0]}-01-01")
            blob_end = pd.Timestamp(f"{BLOB_YEARS[-1]}-12-31")
            ax1.axvspan(blob_start, blob_end, alpha=0.1, color='red', label='Blob Period')
            
            # Set labels and title for top plot
            ax1.set_ylabel(f"{variable_title} ({units})")
            ax1.set_title(f"{variable_title} Timeseries (MHW Mask Approach) - Threshold {min_threshold*100:.0f}%")
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Middle plot: Data values with climatology (Whole region)
            ax2.plot(whole_region_dates, whole_region_data_values, 'g-', linewidth=2, label='Data (Whole region)')
            ax2.plot(whole_region_dates, whole_region_clim_values, 'm--', linewidth=1.5, label='Climatology (Whole region)')
            
            # Mark blob period with vertical spans
            ax2.axvspan(blob_start, blob_end, alpha=0.1, color='red')
            
            # Set labels and title for middle plot
            ax2.set_ylabel(f"{variable_title} ({units})")
            ax2.set_title(f"{variable_title} Timeseries (Whole Region Approach)")
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # Bottom plot: Anomalies comparison
            ax3.plot(mhw_dates, mhw_anomaly_values, 'b-', linewidth=1.5, label='MHW Mask Anomaly')
            ax3.plot(whole_region_dates, whole_region_anomaly_values, 'g-', linewidth=1.5, label='Whole Region Anomaly')
            ax3.axhline(y=0, color='grey', linestyle='-', alpha=0.5)
            
            # Mark blob period with vertical spans
            ax3.axvspan(blob_start, blob_end, alpha=0.1, color='red')
            
            # Set labels for bottom plot
            ax3.set_ylabel(f"{variable_title} Anomaly ({units})")
            ax3.set_xlabel("Date")
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
            
            # Format x-axis to focus on April-November period
            ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=ANALYSIS_MONTHS))
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
            
            # Set x-axis limits to include only Apr-Nov range
            x_min = pd.Timestamp(f"{min(PRE_BLOB_YEARS)}-04-01")
            x_max = pd.Timestamp(f"{max(POST_BLOB_YEARS)}-11-30")
            ax1.set_xlim([x_min, x_max])
            ax2.set_xlim([x_min, x_max])
            ax3.set_xlim([x_min, x_max])
            
            # Add overall title
            plt.suptitle(f"{variable_title} Analysis ({STUDY_REGION['name']}) - Comparing Methods", 
                      fontsize=16, y=0.98)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save figure with threshold in filename
            output_file = os.path.join(OUTPUT_DIR, 
                                    f"{variable_name}_timeseries_method_comparison_threshold_{int(min_threshold*100)}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved method comparison plot to {output_file}")
            
        # Standard plot (as before, but using only MHW mask for blob period)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True, 
                                      gridspec_kw={'height_ratios': [1, 1]})

        # For standard plots, filter data according to the period logic
        plot_df = df.copy()
        if INCLUDE_WHOLE_REGION:
            # During blob period, use MHW mask approach; otherwise use whole region
            plot_df = df[((df['period'] == 'blob') & (df['method'] == 'mhw_mask')) | 
                         ((df['period'] != 'blob') & (df['method'] == 'whole_region'))]
        
        # Extract data for plotting
        plot_df = plot_df.sort_values('date')
        data_values = plot_df['value'].values
        clim_values = plot_df['climatology'].values
        anomaly_values = plot_df['anomaly'].values
        plot_dates = plot_df.index.to_list()
        
        # Top plot: Data values with climatology
        ax1.plot(plot_dates, data_values, 'b-', linewidth=2, label='Data')
        ax1.plot(plot_dates, clim_values, 'r--', linewidth=1.5, label='Climatology')
        
        # Mark blob period with vertical spans
        blob_start = pd.Timestamp(f"{BLOB_YEARS[0]}-01-01")
        blob_end = pd.Timestamp(f"{BLOB_YEARS[-1]}-12-31")
        ax1.axvspan(blob_start, blob_end, alpha=0.1, color='red', label='Blob Period')
        
        # Set labels and title for top plot
        ax1.set_ylabel(f"{variable_title} ({units})")
        ax1.set_title(f"{variable_title} Timeseries - MHW Threshold {min_threshold*100:.0f}%")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Anomalies
        ax2.plot(plot_dates, anomaly_values, 'k-', linewidth=1.5)
        ax2.axhline(y=0, color='grey', linestyle='-', alpha=0.5)
        
        # Mark blob period with vertical spans (same as top plot)
        ax2.axvspan(blob_start, blob_end, alpha=0.1, color='red')
        
        # Set labels for bottom plot
        ax2.set_ylabel(f"{variable_title} Anomaly ({units})")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis to focus on April-November period
        ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=ANALYSIS_MONTHS))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        
        # Set x-axis limits to include only Apr-Nov range
        x_min = pd.Timestamp(f"{min(PRE_BLOB_YEARS)}-04-01")
        x_max = pd.Timestamp(f"{max(POST_BLOB_YEARS)}-11-30")
        ax1.set_xlim([x_min, x_max])
        ax2.set_xlim([x_min, x_max])
        
        # Add overall title
        plt.suptitle(f"{variable_title} Analysis ({STUDY_REGION['name']}) - Threshold {min_threshold*100:.0f}%", 
                   fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure with threshold in filename
        output_file = os.path.join(OUTPUT_DIR, 
                                 f"{variable_name}_timeseries_apr_nov_threshold_{int(min_threshold*100)}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved plot to {output_file}")
        
        # Save the DataFrame to CSV for later analysis
        if INCLUDE_WHOLE_REGION:
            csv_output = os.path.join(OUTPUT_DIR, 
                                    f"{variable_name}_data_with_method_comparison_threshold_{int(min_threshold*100)}.csv")
            df.to_csv(csv_output)
            print(f"  ✓ Saved data to {csv_output}")
    
    # Restore original threshold array
    MHW_THRESHOLD_MIN = original_threshold_array
    print(f"✓ Completed processing {variable_title} timeseries for all thresholds")


def extract_anomalies_to_csv(bool_ds):
    """
    Extract anomalies for all variables and thresholds to a CSV file.
    Only including April-November months.
    For blob period, only includes actual points inside the MHW mask.
    """
    global MHW_THRESHOLD_MIN
    
    print("\n" + "-" * 80)
    print("EXTRACTING RAW ANOMALIES INSIDE MASKS TO CSV FILE (APR-NOV ONLY)")
    print("-" * 80)
    
    # Store original threshold array
    original_threshold_array = MHW_THRESHOLD_MIN.copy() if isinstance(MHW_THRESHOLD_MIN, list) else [MHW_THRESHOLD_MIN]
    
    # Dictionary to hold all anomaly data
    all_anomalies = {
        'date': [],
        'variable': [], 
        'threshold': [],
        'anomaly': [],
        'period': [],
        'lat': [],
        'lon': []
    }
    
    # Process each variable
    for variable_config in VARIABLES:
        variable_name = variable_config['name']
        variable_title = variable_config.get('title', variable_name)
        print(f"Extracting anomalies for {variable_title}...")
        
        # Initialize pre_post_data for this variable
        pre_post_data = {
            'pre_blob': {},
            'post_blob': {}
        }
        
        # Process Pre-Blob period (2011-2013)
        print(f"  Calculating pre-blob data for {variable_name}")
        for year in PRE_BLOB_YEARS:
            for month in ANALYSIS_MONTHS:
                stats, clim_stats = process_domain_month(year, month, variable_config)
                if stats is not None and clim_stats is not None:
                    date_key = f"{year}-{month}"
                    pre_post_data['pre_blob'][date_key] = {
                        'date': stats['dates'],
                        'anomaly': stats['daily_means'] - clim_stats['daily_means'][:len(stats['daily_means'])]
                    }
        
        # Process Post-Blob period (2017-2021)
        print(f"  Calculating post-blob data for {variable_name}")
        for year in POST_BLOB_YEARS:
            for month in ANALYSIS_MONTHS:
                stats, clim_stats = process_domain_month(year, month, variable_config)
                if stats is not None and clim_stats is not None:
                    date_key = f"{year}-{month}"
                    pre_post_data['post_blob'][date_key] = {
                        'date': stats['dates'],
                        'anomaly': stats['daily_means'] - clim_stats['daily_means'][:len(stats['daily_means'])]
                    }
        
        # Process each threshold
        for threshold in original_threshold_array:
            threshold_value = float(threshold)  # Convert to float for consistency
            print(f"  Processing threshold {threshold_value*100:.0f}%")
            
            # Add pre-blob anomalies (same for all thresholds)
            for date_key, data in pre_post_data['pre_blob'].items():
                for i, date in enumerate(data['date']):
                    all_anomalies['date'].append(date)
                    all_anomalies['variable'].append(variable_name)
                    all_anomalies['threshold'].append(threshold_value)
                    all_anomalies['anomaly'].append(data['anomaly'][i])
                    all_anomalies['period'].append('pre_blob')
                    all_anomalies['lat'].append(None)  # No specific lat/lon for domain averages
                    all_anomalies['lon'].append(None)
            
            # Add post-blob anomalies (same for all thresholds)
            for date_key, data in pre_post_data['post_blob'].items():
                for i, date in enumerate(data['date']):
                    all_anomalies['date'].append(date)
                    all_anomalies['variable'].append(variable_name)
                    all_anomalies['threshold'].append(threshold_value)
                    all_anomalies['anomaly'].append(data['anomaly'][i])
                    all_anomalies['period'].append('post_blob')
                    all_anomalies['lat'].append(None)  # No specific lat/lon for domain averages
                    all_anomalies['lon'].append(None)
            
            # Set threshold for blob period processing
            MHW_THRESHOLD_MIN = threshold
            
            # Process Blob period (2014-2016) with domain-averaged approach to be consistent
            print(f"  Processing blob period with threshold {threshold_value*100:.0f}%")
            for year in BLOB_YEARS:
                for month in ANALYSIS_MONTHS:
                    stats, clim_stats = process_blob_month(bool_ds, year, month, variable_config)
                    if stats is not None and clim_stats is not None:
                        for i, date in enumerate(stats['dates']):
                            if i < len(stats['daily_means']) and i < len(clim_stats['daily_means']):
                                anomaly = stats['daily_means'][i] - clim_stats['daily_means'][i]
                                all_anomalies['date'].append(date)
                                all_anomalies['variable'].append(variable_name)
                                all_anomalies['threshold'].append(threshold_value)
                                all_anomalies['anomaly'].append(anomaly)
                                all_anomalies['period'].append('blob')
                                all_anomalies['lat'].append(None)  # Match format with pre/post blob
                                all_anomalies['lon'].append(None)  # Match format with pre/post blob
    
    # Restore original threshold array
    MHW_THRESHOLD_MIN = original_threshold_array
    
    # Create dataframe from collected data
    anomalies_df = pd.DataFrame(all_anomalies)
    
    # Sort by date, variable, and threshold
    anomalies_df = anomalies_df.sort_values(['date', 'variable', 'threshold'])
    
    # Define output file path
    output_file = os.path.join(OUTPUT_DIR, 'all_variable_anomalies_inside_mask.csv')
    
    # Save to CSV
    anomalies_df.to_csv(output_file, index=False)
    
    print(f"✓ Extracted {len(anomalies_df)} anomaly records to {output_file}")
    
    # Create boxplot of anomalies by variable and threshold
    plot_anomaly_boxplots(anomalies_df)

# Helper function to calculate climatology field
def calculate_climatology_field(clim_datasets, variable_name, depth_idx, month):
    """Calculate climatology field for a specific variable, depth, and month."""
    # Collect fields from all years
    all_fields = []
    all_shapes = []  # Track all shapes to find the most common one
    
    for year, ds in clim_datasets.items():
        if variable_name not in ds:
            continue
            
        var_data = ds[variable_name]
        
        if 'time' in var_data.dims:
            # Average all time steps in this month
            month_data = var_data.isel(depth=depth_idx).mean(dim='time').values
        else:
            month_data = var_data.isel(depth=depth_idx).values
        
        all_fields.append(month_data)
        all_shapes.append(month_data.shape)
    
    if not all_fields:
        return None
    
    # Find most common shape to standardize on
    from collections import Counter
    shape_counts = Counter(all_shapes)
    most_common_shape = shape_counts.most_common(1)[0][0]
    
    # Resize fields to the most common shape
    standardized_fields = []
    for field in all_fields:
        if field.shape != most_common_shape:
            # Trim to the smaller dimensions
            min_rows = min(field.shape[0], most_common_shape[0])
            min_cols = min(field.shape[1], most_common_shape[1])
            standardized_fields.append(field[:min_rows, :min_cols])
        else:
            standardized_fields.append(field)
    
    # Use the smallest common dimensions for the final result
    min_rows = min([field.shape[0] for field in standardized_fields])
    min_cols = min([field.shape[1] for field in standardized_fields])
    
    # Trim all fields to these common dimensions
    trimmed_fields = [field[:min_rows, :min_cols] for field in standardized_fields]
    
    # Stack and average across years
    return np.nanmean(np.stack(trimmed_fields), axis=0)

# Function to create boxplots of anomalies
def plot_anomaly_boxplots(df):
    """Create boxplots of anomalies by variable and threshold for the blob period."""
    print("Creating boxplots of anomalies...")
    
    # Filter only blob period data
    blob_df = df[df['period'] == 'blob'].copy()
    
    if blob_df.empty:
        print("No blob period data to plot")
        return
        
    # Get unique variables and thresholds
    variables = blob_df['variable'].unique()
    thresholds = sorted(blob_df['threshold'].unique())
    
    # Create plots for each variable
    for variable_name in variables:
        var_df = blob_df[blob_df['variable'] == variable_name]
        
        # Get variable title and units
        var_title = next((v['title'] for v in VARIABLES if v['name'] == variable_name), variable_name)
        if variable_name == 'temp':
            units = "°C"
        elif variable_name == 'TOT_PROD':
            units = "mol/m³/y"
        elif variable_name == 'POC_FLUX_IN':
            units = "mol/m²/y"
        else:
            units = ""
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Prepare data for boxplot
        boxplot_data = [var_df[var_df['threshold'] == t]['anomaly'].values for t in thresholds]
        
        # Create boxplot
        bp = plt.boxplot(boxplot_data, patch_artist=True, notch=True)
        
        # Customize colors
        colors = ['skyblue', 'lightgreen', 'salmon']
        for i, box in enumerate(bp['boxes']):
            box.set(facecolor=colors[i % len(colors)])
        
        # Add number of data points as text
        for i, t in enumerate(thresholds):
            n_points = len(var_df[var_df['threshold'] == t])
            plt.text(i+1, plt.ylim()[0]*0.9, f"n={n_points}", 
                    horizontalalignment='center', size='small')
            
        # Set labels and title
        plt.xlabel('MHW Threshold')
        plt.ylabel(f'Anomaly ({units})')
        plt.title(f'{var_title} Anomaly Distribution (Blob Period 2014-2016)')
        
        # Set x-tick labels to threshold percentages
        plt.xticks(range(1, len(thresholds)+1), [f"{int(t*100)}%" for t in thresholds])
        
        # Add zero line
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add grid
        plt.grid(True, axis='y', alpha=0.3)
        
        # Save figure
        output_file = os.path.join(OUTPUT_DIR, f'{variable_name}_anomaly_boxplot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved boxplot to {output_file}")

def plot_seasonal_cycle(variable_config, bool_ds):
    """
    Create a seasonal cycle plot where April follows November directly,
    showing the continuous growing season pattern without winter gaps.
    """
    global MHW_THRESHOLD_MIN
    
    variable_name = variable_config['name']
    variable_title = variable_config['title']
    
    # Define units based on variable
    if variable_name == 'temp':
        units = "°C"
    elif variable_name == 'TOT_PROD':
        units = "mol/m³/y"
    elif variable_name == 'POC_FLUX_IN':
        units = "molC/m²/y"
    else:
        units = ""
        
    print(f"\nCreating seasonal cycle plot for {variable_title}...")
    
    # Month order for x-axis: Apr-Nov
    months = [4, 5, 6, 7, 8, 9, 10, 11]
    month_labels = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
    
    # Data containers
    monthly_data = {
        'pre_blob': {m: {'values': [], 'climatology': [], 'anomaly': []} for m in months},
        'blob': {m: {'values': [], 'climatology': [], 'anomaly': []} for m in months},
        'post_blob': {m: {'values': [], 'climatology': [], 'anomaly': []} for m in months}
    }
    
    # Process each time period
    periods = {
        'pre_blob': PRE_BLOB_YEARS,
        'blob': BLOB_YEARS,
        'post_blob': POST_BLOB_YEARS
    }
    
    # Process Pre-Blob and Post-Blob (whole domain) periods
    for period_name, years in periods.items():
        if period_name in ['pre_blob', 'post_blob']:
            for year in years:
                for month in months:
                    stats, clim_stats = process_domain_month(year, month, variable_config)
                    if stats is not None and clim_stats is not None:
                        # Calculate monthly averages
                        monthly_data[period_name][month]['values'].append(np.nanmean(stats['daily_means']))
                        monthly_data[period_name][month]['climatology'].append(np.nanmean(clim_stats['daily_means']))
                        monthly_data[period_name][month]['anomaly'].append(
                            np.nanmean(stats['daily_means']) - np.nanmean(clim_stats['daily_means'])
                        )
    
    # Process Blob period (using MHW masks)
    if bool_ds is not None:
        # Store original threshold to restore later
        original_threshold = MHW_THRESHOLD_MIN
        # Use first threshold if it's a list
        threshold = original_threshold[0] if isinstance(original_threshold, list) else original_threshold
        
        for year in BLOB_YEARS:
            for month in months:
                stats, clim_stats = process_blob_month(bool_ds, year, month, variable_config)
                if stats is not None and clim_stats is not None:
                    # Calculate monthly averages
                    monthly_data['blob'][month]['values'].append(np.nanmean(stats['daily_means']))
                    monthly_data['blob'][month]['climatology'].append(np.nanmean(clim_stats['daily_means']))
                    monthly_data['blob'][month]['anomaly'].append(
                        np.nanmean(stats['daily_means']) - np.nanmean(clim_stats['daily_means'])
                    )
        
        # Restore original threshold
        MHW_THRESHOLD_MIN = original_threshold
    
    # Calculate mean values for each month and period
    monthly_means = {
        'pre_blob': [np.nanmean(monthly_data['pre_blob'][m]['values']) for m in months],
        'blob': [np.nanmean(monthly_data['blob'][m]['values']) for m in months],
        'post_blob': [np.nanmean(monthly_data['post_blob'][m]['values']) for m in months]
    }
    
    monthly_clim = {
        'pre_blob': [np.nanmean(monthly_data['pre_blob'][m]['climatology']) for m in months],
        'blob': [np.nanmean(monthly_data['blob'][m]['climatology']) for m in months],
        'post_blob': [np.nanmean(monthly_data['post_blob'][m]['climatology']) for m in months]
    }
    
    monthly_anom = {
        'pre_blob': [np.nanmean(monthly_data['pre_blob'][m]['anomaly']) for m in months],
        'blob': [np.nanmean(monthly_data['blob'][m]['anomaly']) for m in months],
        'post_blob': [np.nanmean(monthly_data['post_blob'][m]['anomaly']) for m in months]
    }
    
    # Create the figure with two stacked plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=150)
    
    # Plot monthly means and climatology on top panel
    x = np.arange(len(months))
    
    # Pre-blob period
    ax1.plot(x, monthly_means['pre_blob'], 'o-', color='blue', label='Pre-Blob (2011-2013)')
    ax1.plot(x, monthly_clim['pre_blob'], '--', color='blue', alpha=0.5)
    
    # Blob period
    ax1.plot(x, monthly_means['blob'], 'o-', color='red', label='Blob (2014-2016)')
    ax1.plot(x, monthly_clim['blob'], '--', color='red', alpha=0.5)
    
    # Post-blob period
    ax1.plot(x, monthly_means['post_blob'], 'o-', color='green', label='Post-Blob (2017-2021)')
    ax1.plot(x, monthly_clim['post_blob'], '--', color='green', alpha=0.5)
    
    # Add climatology to legend
    ax1.plot([], [], '--', color='gray', label='Climatology')
    
    # Configure top plot
    ax1.set_xticks(x)
    ax1.set_xticklabels(month_labels)
    ax1.set_xlabel('Month')
    ax1.set_ylabel(f'{variable_title} ({units})')
    ax1.set_title(f'Seasonal Cycle of {variable_title}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot anomalies on bottom panel
    bar_width = 0.25
    
    # Plot anomaly bars
    ax2.bar(x - bar_width, monthly_anom['pre_blob'], width=bar_width, 
            color='blue', alpha=0.7, label='Pre-Blob (2011-2013)')
    ax2.bar(x, monthly_anom['blob'], width=bar_width,
            color='red', alpha=0.7, label='Blob (2014-2016)')
    ax2.bar(x + bar_width, monthly_anom['post_blob'], width=bar_width,
            color='green', alpha=0.7, label='Post-Blob (2017-2021)')
    
    # Add zero line for anomalies
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Configure bottom plot
    ax2.set_xticks(x)
    ax2.set_xticklabels(month_labels)
    ax2.set_xlabel('Month')
    ax2.set_ylabel(f'Anomaly ({units})')
    ax2.set_title(f'{variable_title} Anomalies by Period')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # Add depth information as text annotation
    if variable_config['type'] == 'depth_averaged':
        depth_info = f"0-50m average"
    else:  # single_depth
        depth_info = f"{abs(variable_config['target_depth'])}m depth"
    
    ax1.text(0.02, 0.95, depth_info, transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Add overall title and adjust layout
    plt.suptitle(f'Seasonal Cycle Analysis: {variable_title}', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    threshold_str = ""
    if isinstance(original_threshold, list):
        threshold_str = f"_threshold_{int(original_threshold[0]*100)}"
        
    output_file = os.path.join(OUTPUT_DIR, f'{variable_name}_seasonal_cycle{threshold_str}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved seasonal cycle plot to {output_file}")

def plot_threshold_comparison():
    """Create comparison plots showing anomalies for all thresholds overlaid, including whole study region."""
    print("\n" + "-" * 80)
    print("CREATING THRESHOLD COMPARISON PLOTS (INCLUDING WHOLE REGION)")
    print("-" * 80)
    
    # Define colors for different thresholds
    colors = {
        0.0: 'black',
        0.4: 'green',
        0.8: 'red'
    }
    
    # Read the anomaly CSV files
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*_data_with_method_comparison_threshold_*.csv'))
    if not csv_files:
        print("Error: No anomaly files found")
        return
    
    # Process each variable
    for variable_config in VARIABLES:
        variable_name = variable_config['name']
        variable_title = variable_config['title']
        
        # Define units based on variable
        if variable_name == 'temp':
            units = "°C"
        elif variable_name == 'TOT_PROD':
            units = "mol/m³/y"
        elif variable_name == 'POC_FLUX_IN':
            units = "mol/m²/y"
        else:
            units = ""
        
        # Find CSV files for this variable
        var_csv_files = [f for f in csv_files if variable_name in f]
        if not var_csv_files:
            print(f"No data files found for {variable_name}")
            continue
            
        print(f"Creating threshold comparison plot for {variable_title} (including whole region)...")
        
        # Load data from all thresholds
        combined_df = pd.DataFrame()
        for csv_file in var_csv_files:
            # Extract threshold from filename
            threshold_match = re.search(r'threshold_(\d+)\.csv', csv_file)
            if threshold_match:
                threshold = int(threshold_match.group(1)) / 100.0
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'])
                df['threshold'] = threshold
                combined_df = pd.concat([combined_df, df])
        
        if combined_df.empty:
            print(f"No data loaded for {variable_name}")
            continue
        
        # Create figure for threshold comparison
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Filter data for blob period only
        blob_df = combined_df[combined_df['period'] == 'blob'].copy()
        
        # First, plot the whole region line (will be a reference line)
        whole_region_data = blob_df[blob_df['method'] == 'whole_region'].sort_values('date')
        if not whole_region_data.empty:
            ax.plot(whole_region_data['date'], whole_region_data['anomaly'],
                   label="Whole Study Region",
                   color='black', linewidth=3, linestyle='-')
        
        # Then plot each MHW threshold mask
        for threshold in sorted(blob_df[blob_df['method'] == 'mhw_mask']['threshold'].unique()):
            # Skip if it's 0 as we're already plotting whole region
            if threshold == 0.0:
                continue
                
            # MHW mask approach
            mask_data = blob_df[(blob_df['threshold'] == threshold) & 
                               (blob_df['method'] == 'mhw_mask')].sort_values('date')
            if not mask_data.empty:
                ax.plot(mask_data['date'], mask_data['anomaly'], 
                       label=f"MHW Mask {int(threshold*100)}%",
                       color=colors.get(threshold, 'gray'), 
                       linewidth=2)
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Labels and title
        ax.set_title(f"{variable_title} Anomalies: Comparing MHW Thresholds vs Whole Region")
        ax.set_ylabel(f"Anomaly ({units})")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=ANALYSIS_MONTHS))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        
        # Set x-axis limits to include only blob period
        x_min = pd.Timestamp(f"{min(BLOB_YEARS)}-04-01")
        x_max = pd.Timestamp(f"{max(BLOB_YEARS)}-11-30")
        ax.set_xlim([x_min, x_max])
        
        # Add depth information as text annotation
        if variable_config['type'] == 'depth_averaged':
            depth_info = f"0-50m average"
        else:  # single_depth
            depth_info = f"{abs(variable_config['target_depth'])}m depth"
            
        ax.text(0.02, 0.95, depth_info, transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Save figure
        output_file = os.path.join(OUTPUT_DIR, f'{variable_name}_threshold_comparison_with_wholeregion.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved threshold comparison to {output_file}")
        
        # Create bar chart comparing mean anomalies
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate mean anomalies for each threshold during blob period
        mean_anomalies = []
        std_anomalies = []
        labels = ["Whole Region"]  # Start with whole region
        
        # First get whole region mean
        whole_region_mean = whole_region_data['anomaly'].mean()
        whole_region_std = whole_region_data['anomaly'].std() / 2  # Using half std for error bars
        mean_anomalies.append(whole_region_mean)
        std_anomalies.append(whole_region_std)
        
        # Then get each threshold mean
        for threshold in sorted(blob_df[blob_df['method'] == 'mhw_mask']['threshold'].unique()):
            # Skip if it's 0 as we're already plotting whole region
            if threshold == 0.0:
                continue
                
            mask_data = blob_df[(blob_df['threshold'] == threshold) & 
                               (blob_df['method'] == 'mhw_mask')]
            
            if not mask_data.empty:
                mean_anomalies.append(mask_data['anomaly'].mean())
                std_anomalies.append(mask_data['anomaly'].std() / 2)
                labels.append(f"{int(threshold*100)}%")
        
        # Plot bars
        bars = ax.bar(range(len(mean_anomalies)), mean_anomalies, 
                     yerr=std_anomalies, capsize=5, 
                     color=['black'] + [colors.get(float(int(label[:-1])/100), 'gray') for label in labels[1:]])
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Set x-ticks and labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        
        # Add labels and title
        ax.set_xlabel("MHW Threshold")
        ax.set_ylabel(f"Mean Anomaly ({units})")
        ax.set_title(f"Mean {variable_title} Anomalies by Threshold (Blob Period 2014-2016)")
        ax.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            offset = 0.2 if height > 0 else -0.6
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                   f'{height:.2f}',
                   ha='center', va='bottom', rotation=0)
        
        # Save bar chart
        output_file = os.path.join(OUTPUT_DIR, f'{variable_name}_mean_anomalies_by_threshold.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved mean anomalies comparison to {output_file}")
    
    print("✓ Completed threshold comparison plots with whole region")

# Add this function after the plot_threshold_comparison function
def plot_comprehensive_variable_summary(variable_config, bool_ds):
    """
    Create a minimal summary plot for each variable showing thresholds comparison.
    """
    print("\n" + "-" * 80)
    print(f"CREATING MINIMAL SUMMARY FOR {variable_config['title']}")
    print("-" * 80)
    
    variable_name = variable_config['name']
    variable_title = variable_config['title']
    
    # Define units based on variable
    if variable_name == 'temp':
        units = "°C"
    elif variable_name == 'TOT_PROD':
        units = "mol/m³/y"
    elif variable_name == 'POC_FLUX_IN':
        units = "mol/m²/y"
    else:
        units = ""
    
    # SIMPLIFY: Print blob period monthly averages to console
    print(f"Monthly anomalies for {variable_title} during blob period:")
    
    # Simply print the data for each threshold
    thresholds = MHW_THRESHOLD_MIN if isinstance(MHW_THRESHOLD_MIN, list) else [MHW_THRESHOLD_MIN]
    for threshold in thresholds:
        print(f"  Threshold {int(threshold*100)}%:")
        
        # Process blob data for this threshold
        MHW_THRESHOLD_MIN_save = MHW_THRESHOLD_MIN
        MHW_THRESHOLD_MIN = threshold
        
        monthly_data = {month: [] for month in ANALYSIS_MONTHS}
        
        for year in BLOB_YEARS:
            for month in ANALYSIS_MONTHS:
                stats, clim_stats = process_blob_month(bool_ds, year, month, variable_config)
                if stats is not None and clim_stats is not None:
                    anomalies = stats['daily_means'] - clim_stats['daily_means'][:len(stats['daily_means'])]
                    monthly_data[month].extend(anomalies)
        
        # Restore threshold
        MHW_THRESHOLD_MIN = MHW_THRESHOLD_MIN_save
        
        # Print monthly averages
        for month in sorted(monthly_data.keys()):
            if monthly_data[month]:
                month_name = pd.Timestamp(2020, month, 1).strftime('%B')
                avg_anomaly = np.nanmean(monthly_data[month])
                print(f"    {month_name}: {avg_anomaly:.2f} {units} (from {len(monthly_data[month])} days)")
    
    # Create a very simple bar chart of mean anomalies only
    # This won't try to save any complex visuals that cause memory issues
    try:
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        means = []
        labels = []
        
        # Get means for each threshold
        for threshold in thresholds:
            MHW_THRESHOLD_MIN_save = MHW_THRESHOLD_MIN
            MHW_THRESHOLD_MIN = threshold
            
            all_anomalies = []
            for year in BLOB_YEARS:
                for month in ANALYSIS_MONTHS:
                    stats, clim_stats = process_blob_month(bool_ds, year, month, variable_config)
                    if stats is not None and clim_stats is not None:
                        anomalies = stats['daily_means'] - clim_stats['daily_means'][:len(stats['daily_means'])]
                        all_anomalies.extend(anomalies)
            
            MHW_THRESHOLD_MIN = MHW_THRESHOLD_MIN_save
            
            if all_anomalies:
                means.append(np.nanmean(all_anomalies))
                labels.append(f"{int(threshold*100)}%")
        
        # Simple bar chart
        ax.bar(labels, means)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_ylabel(f"Mean Anomaly ({units})")
        ax.set_title(f"{variable_title} Mean Anomalies by Threshold")
        
        # Save simple figure
        output_file = os.path.join(OUTPUT_DIR, f'{variable_name}_mean_anomalies_simple.png')
        plt.savefig(output_file, dpi=100)  # Lower DPI to prevent size issues
        plt.close()
        print(f"  ✓ Saved simple summary to {output_file}")
    
    except Exception as e:
        print(f"  × Warning: Could not create simple plot: {e}")
        # Continue execution even if plotting fails

def plot_consolidated_thresholds(variable_config, bool_ds):
    """
    Create a single consolidated plot showing all thresholds for a variable,
    with improved x-axis formatting that compresses winter months.
    """
    global MHW_THRESHOLD_MIN  # Add this global declaration
    
    print(f"\nCreating consolidated threshold plot for {variable_config['title']}...")
    
    variable_name = variable_config['name']
    variable_title = variable_config['title']
    
    # Define units based on variable
    if variable_name == 'temp':
        units = "°C"
    elif variable_name == 'TOT_PROD':
        units = "mol/m³/y"
    elif variable_name == 'POC_FLUX_IN':
        units = "mol/m²/y"
    else:
        units = ""
    
    # Store original threshold value to restore later
    original_threshold_array = MHW_THRESHOLD_MIN.copy() if isinstance(MHW_THRESHOLD_MIN, list) else MHW_THRESHOLD_MIN
    thresholds = original_threshold_array if isinstance(original_threshold_array, list) else [original_threshold_array]
    
    # Process whole region data first (for all periods)
    whole_region_data = {
        'dates': [],
        'anomalies': [],
        'period': []
    }
    
    # Pre-blob period
    for year in PRE_BLOB_YEARS:
        for month in ANALYSIS_MONTHS:
            stats, clim_stats = process_domain_month(year, month, variable_config)
            if stats is not None and clim_stats is not None:
                anomalies = stats['daily_means'] - clim_stats['daily_means'][:len(stats['daily_means'])]
                whole_region_data['dates'].extend(stats['dates'])
                whole_region_data['anomalies'].extend(anomalies)
                whole_region_data['period'].extend(['pre_blob'] * len(stats['dates']))
    
    # Blob period
    for year in BLOB_YEARS:
        for month in ANALYSIS_MONTHS:
            stats, clim_stats = process_domain_month(year, month, variable_config)
            if stats is not None and clim_stats is not None:
                anomalies = stats['daily_means'] - clim_stats['daily_means'][:len(stats['daily_means'])]
                whole_region_data['dates'].extend(stats['dates'])
                whole_region_data['anomalies'].extend(anomalies)
                whole_region_data['period'].extend(['blob'] * len(stats['dates']))
    
    # Post-blob period
    for year in POST_BLOB_YEARS:
        for month in ANALYSIS_MONTHS:
            stats, clim_stats = process_domain_month(year, month, variable_config)
            if stats is not None and clim_stats is not None:
                anomalies = stats['daily_means'] - clim_stats['daily_means'][:len(stats['daily_means'])]
                whole_region_data['dates'].extend(stats['dates'])
                whole_region_data['anomalies'].extend(anomalies)
                whole_region_data['period'].extend(['post_blob'] * len(stats['dates']))
    
    # Process each threshold for the blob period
    threshold_data = {}
    for threshold in thresholds:
        MHW_THRESHOLD_MIN = threshold
        threshold_data[threshold] = {
            'dates': [],
            'anomalies': []
        }
        
        for year in BLOB_YEARS:
            for month in ANALYSIS_MONTHS:
                stats, clim_stats = process_blob_month(bool_ds, year, month, variable_config)
                if stats is not None and clim_stats is not None:
                    anomalies = stats['daily_means'] - clim_stats['daily_means'][:len(stats['daily_means'])]
                    threshold_data[threshold]['dates'].extend(stats['dates'])
                    threshold_data[threshold]['anomalies'].extend(anomalies)
    
    # Restore original threshold
    MHW_THRESHOLD_MIN = original_threshold_array
    
    # Create figure with improved x-axis
    fig, ax = plt.figure(figsize=(16, 8)), plt.gca()
    
    # Sort whole region data by date
    wr_df = pd.DataFrame(whole_region_data).sort_values('dates')
    
    # Plot whole region data
    ax.plot(wr_df['dates'], wr_df['anomalies'], 'k-', linewidth=2.5, label='Whole Region')
    
    # Define colors for different thresholds
    cmap = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    
    # Plot each threshold (blob period only)
    for i, threshold in enumerate(thresholds):
        # Skip threshold=0 as it's equivalent to the whole region
        if threshold == 0.0:
            continue
            
        # Sort by date
        if threshold in threshold_data:
            th_df = pd.DataFrame({
                'dates': threshold_data[threshold]['dates'],
                'anomalies': threshold_data[threshold]['anomalies']
            }).sort_values('dates')
            
            if not th_df.empty:
                # Use more distinct line styles for clarity
                line_style = '-'
                if i > 2:
                    line_style = '--'
                if i > 4:
                    line_style = '-.'
                    
                ax.plot(
                    th_df['dates'], 
                    th_df['anomalies'],
                    line_style,
                    color=cmap[i],
                    linewidth=2,
                    label=f'Threshold {int(threshold*100)}%'
                )
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Mark blob period with vertical spans
    blob_start = pd.Timestamp(f"{BLOB_YEARS[0]}-04-01")
    blob_end = pd.Timestamp(f"{BLOB_YEARS[-1]}-11-30")
    ax.axvspan(blob_start, blob_end, alpha=0.1, color='red', label='Blob Period')
    
    # Custom x-axis formatting to compress winter months and show only specified months
    years = list(range(min(PRE_BLOB_YEARS), max(POST_BLOB_YEARS) + 1))
    months_to_show = [4, 6, 8, 10]  # April, June, August, October
    
    # Create custom tick positions
    tick_positions = []
    tick_labels = []
    
    for year in years:
        for month in sorted(ANALYSIS_MONTHS):
            if month in months_to_show:
                date = pd.Timestamp(f"{year}-{month:02d}-15")
                tick_positions.append(date)
                
                # Use abbreviated month names
                month_name = date.strftime('%b')
                if month == 4:  # Add year with April
                    tick_labels.append(f"{month_name}\n{year}")
                else:
                    tick_labels.append(f"{month_name}")
    
    # Set custom ticks
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # Format x-axis limits to include only Apr-Nov range
    x_min = pd.Timestamp(f"{min(PRE_BLOB_YEARS)}-04-01")
    x_max = pd.Timestamp(f"{max(POST_BLOB_YEARS)}-11-30")
    ax.set_xlim([x_min, x_max])
    
    # Add depth information
    if variable_config['type'] == 'depth_averaged':
        depth_info = f"0-50m average"
    else:  # single_depth
        depth_info = f"{abs(variable_config['target_depth'])}m depth"
        
    ax.text(0.02, 0.95, depth_info, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Add labels and title
    ax.set_title(f"{variable_title} Anomalies: All Thresholds Comparison", fontsize=14)
    ax.set_ylabel(f"Anomaly ({units})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add text showing mean anomalies for each threshold during blob period
    text_str = "Mean Blob Period Anomalies:\n"
    text_str += f"Whole Region: {np.nanmean(wr_df[wr_df['period'] == 'blob']['anomalies']):.2f} {units}\n"
    
    for threshold in thresholds:
        if threshold == 0.0:
            continue
            
        if threshold in threshold_data and threshold_data[threshold]['anomalies']:
            mean_val = np.nanmean(threshold_data[threshold]['anomalies'])
            text_str += f"Threshold {int(threshold*100)}%: {mean_val:.2f} {units}\n"
    
    # Add text box with mean values
    ax.text(0.02, 0.05, text_str, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(OUTPUT_DIR, f'{variable_name}_all_thresholds_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved consolidated thresholds plot to {output_file}")

def export_all_variable_anomalies(bool_ds):
    """
    Export anomalies for all variables and thresholds to a CSV file.
    This includes all periods (pre_blob, blob, post_blob) and all thresholds.
    """
    print("\nExporting complete timeseries anomalies for all variables and thresholds...")
    global MHW_THRESHOLD_MIN  # Add this global declaration
    
    # Store original threshold array
    original_threshold_array = MHW_THRESHOLD_MIN.copy() if isinstance(MHW_THRESHOLD_MIN, list) else [MHW_THRESHOLD_MIN]
    
    # Dictionary to hold all anomaly data
    all_anomalies = {
        'date': [],
        'variable': [], 
        'threshold': [],
        'anomaly': [],
        'period': []
    }
    
    # Process each variable
    for variable_config in VARIABLES:
        variable_name = variable_config['name']
        print(f"Processing {variable_name}...")
        
        # Process pre-blob and post-blob periods (same for all thresholds)
        pre_blob_data = {}
        post_blob_data = {}
        
        # Pre-blob period
        print(f"  Calculating pre-blob data...")
        for year in PRE_BLOB_YEARS:
            for month in ANALYSIS_MONTHS:
                stats, clim_stats = process_domain_month(year, month, variable_config)
                if stats is not None and clim_stats is not None:
                    for i, date in enumerate(stats['dates']):
                        anomaly = stats['daily_means'][i] - clim_stats['daily_means'][i] if i < len(clim_stats['daily_means']) else np.nan
                        date_str = date.strftime('%Y-%m-%d')
                        pre_blob_data[date_str] = anomaly
        
        # Post-blob period
        print(f"  Calculating post-blob data...")
        for year in POST_BLOB_YEARS:
            for month in ANALYSIS_MONTHS:
                stats, clim_stats = process_domain_month(year, month, variable_config)
                if stats is not None and clim_stats is not None:
                    for i, date in enumerate(stats['dates']):
                        anomaly = stats['daily_means'][i] - clim_stats['daily_means'][i] if i < len(clim_stats['daily_means']) else np.nan
                        date_str = date.strftime('%Y-%m-%d')
                        post_blob_data[date_str] = anomaly
        
        # Process each threshold for the blob period
        for threshold in original_threshold_array:
            threshold_value = float(threshold)
            print(f"  Processing threshold {threshold_value*100:.0f}%")
            
            # Add pre-blob anomalies (same for all thresholds)
            for date_str, anomaly in pre_blob_data.items():
                all_anomalies['date'].append(date_str)
                all_anomalies['variable'].append(variable_name)
                all_anomalies['threshold'].append(threshold_value)
                all_anomalies['anomaly'].append(anomaly)
                all_anomalies['period'].append('pre_blob')
            
            # Add post-blob anomalies (same for all thresholds)
            for date_str, anomaly in post_blob_data.items():
                all_anomalies['date'].append(date_str)
                all_anomalies['variable'].append(variable_name)
                all_anomalies['threshold'].append(threshold_value)
                all_anomalies['anomaly'].append(anomaly)
                all_anomalies['period'].append('post_blob')
            
            # Set threshold for blob period processing
            MHW_THRESHOLD_MIN = threshold
            
            # Process blob period with the current threshold
            print(f"  Processing blob period with threshold {threshold_value*100:.0f}%...")
            for year in BLOB_YEARS:
                for month in ANALYSIS_MONTHS:
                    stats, clim_stats = process_blob_month(bool_ds, year, month, variable_config)
                    if stats is not None and clim_stats is not None:
                        for i, date in enumerate(stats['dates']):
                            anomaly = stats['daily_means'][i] - clim_stats['daily_means'][i] if i < len(clim_stats['daily_means']) else np.nan
                            date_str = date.strftime('%Y-%m-%d')
                            all_anomalies['date'].append(date_str)
                            all_anomalies['variable'].append(variable_name)
                            all_anomalies['threshold'].append(threshold_value)
                            all_anomalies['anomaly'].append(anomaly)
                            all_anomalies['period'].append('blob')
    
    # Restore original threshold array
    MHW_THRESHOLD_MIN = original_threshold_array
    
    # Create dataframe from collected data
    anomalies_df = pd.DataFrame(all_anomalies)
    
    # Sort by date, variable, and threshold
    anomalies_df = anomalies_df.sort_values(['date', 'variable', 'threshold'])
    
    # Define output file path
    output_file = os.path.join(OUTPUT_DIR, 'all_variable_anomalies.csv')
    
    # Save to CSV
    anomalies_df.to_csv(output_file, index=False)
    
    print(f"✓ Exported {len(anomalies_df)} anomaly records to {output_file}")

def average_depth_stats(depth_stats_list):
    """
    Average statistics across multiple depth levels.
    
    Parameters:
    -----------
    depth_stats_list : list
        List of statistics dictionaries from different depths
        
    Returns:
    --------
    dict
        Averaged statistics
    """
    if not depth_stats_list:
        return None
    
    # Get the first depth's stats structure
    first_stats = depth_stats_list[0]
    
    # Initialize result with same structure
    result = {
        'daily_means': np.zeros_like(first_stats['daily_means']),
        'daily_stds': np.zeros_like(first_stats['daily_stds']),
    }
    
    # Handle the count key which may have different names
    if 'daily_counts' in first_stats:
        count_key = 'daily_counts'
        result['daily_counts'] = np.zeros_like(first_stats['daily_counts'])
    elif 'day_counts' in first_stats:
        count_key = 'day_counts'
        result['day_counts'] = np.zeros_like(first_stats['day_counts'])
    else:
        # Create a default if neither exists
        count_key = None
        result['day_counts'] = np.zeros_like(first_stats['daily_means'])
    
    # Add timestamps if they exist in the first stats
    if 'timestamps' in first_stats:
        result['timestamps'] = first_stats['timestamps']
    
    # Count valid depths for proper averaging
    valid_depths = np.zeros_like(first_stats['daily_means'])
    
    # Sum across depths
    for stats in depth_stats_list:
        valid_mask = ~np.isnan(stats['daily_means'])
        result['daily_means'][valid_mask] += stats['daily_means'][valid_mask]
        result['daily_stds'][valid_mask] += stats['daily_stds'][valid_mask]
        
        # Handle count key flexibly
        if count_key and count_key in stats:
            result[count_key][valid_mask] += stats[count_key][valid_mask]
        
        valid_depths[valid_mask] += 1
    
    # Avoid division by zero
    valid_depths[valid_depths == 0] = 1
    
    # Calculate averages
    result['daily_means'] = result['daily_means'] / valid_depths
    result['daily_stds'] = result['daily_stds'] / valid_depths
    
    # Add dates if they exist in the first stats
    if 'dates' in first_stats:
        result['dates'] = first_stats['dates']
    
    return result

# Add call to this function in the main() function:
def main():
    """Main execution function."""
    print("-" * 80)
    print(f"COMPLETE TIMESERIES ANALYSIS (2011-2021) WITH DEPTH-RESOLVED MHW MASKS")
    print(f"Pre-Blob (2011-2013): Full Domain")
    print(f"Blob (2014-2016): Marine Heatwave Areas (depth-specific masks)")
    print(f"Post-Blob (2017-2021): Full Domain")
    print(f"MONTHS: April (003) through November (011) only")
    print("-" * 80)
    
    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load boolean data (needed for blob period)
    bool_result = load_boolean_data()
    if bool_result is None:
        print("ERROR: Could not load boolean extreme event data")
        return
    
    bool_ds, lons, lats = bool_result
    
    # Process each variable
    for variable_config in VARIABLES:
        process_complete_timeseries(variable_config, bool_ds)
        
        # Add the seasonal cycle plot function
        plot_seasonal_cycle(variable_config, bool_ds)
        
        # Add the consolidated thresholds plot
        plot_consolidated_thresholds(variable_config, bool_ds)
    
    # Extract anomalies to CSV (original function - keep for backward compatibility)
    extract_anomalies_to_csv(bool_ds)
    
    # Add the new function to export all anomalies with all thresholds
    export_all_variable_anomalies(bool_ds)
    
    # Create threshold comparison plots
    plot_threshold_comparison()
    
    print("\nComplete timeseries analysis finished!")

if __name__ == "__main__":
    main()

