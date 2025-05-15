#!/usr/bin/env python3
"""
Create a comprehensive timeseries analysis spanning 2011-2021, including:
1. Pre-Blob (2011-2013): Whole domain statistics
2. Blob (2014-2016): MHW contour mask statistics (using surface layer mask for all depths)
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
OUTPUT_DIR = '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/surface_mask_timeseries_apr_nov/'  # Updated output directory
BOOLEAN_FILE_PATH = '/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/extreme_analysis/boolean_arrays/romsoc_fully_coupled/present/boolean_array_hobday2016_threshold_and_climatology_temp_37zlevs_full_1x1meanpool_downsampling_2011-2021analysisperiod_90perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing.nc'
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
ANALYSIS_MONTHS = range(5, 11)  # 5-10 correspond to May-Oct in the data files

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
MHW_THRESHOLD_MIN = [0.1, 0.3, 0.5]  # Lower bound for contour
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

def get_monthly_extreme_data(bool_ds, year, month):
    """
    Extract monthly extreme event frequency for a specific month of a year.
    Simplified to only use the surface layer (depth_idx=0).
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
    
    # Always use surface layer (depth_idx=0)
    depth_idx = 0
    
    # Get data for this month at the surface layer
    monthly_data = bool_ds[BOOL_VAR_NAME].isel(depth=depth_idx, time=month_indices[0])
    
    # Calculate mean frequency for the month
    monthly_frequency = monthly_data.mean(dim='time')
    
    print(f"Extracted extreme event frequency for {year}-{month:02d} at surface layer with {len(month_indices[0])} time points")
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
    variable_name = variable_config['name']
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
        print(f"Month {month} is outside the May-October analysis window")
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

# %%
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
    Process a single month using MHW contour mask from surface layer.
    Simplified to use only the surface layer mask for all depths.
    """
    # Check if month is in analysis range
    if month not in ANALYSIS_MONTHS:
        return None, None
    
    global MHW_THRESHOLD_MIN  # Proper global declaration
    variable_name = variable_config['name']
    
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
    
    # Get extreme event frequency for surface layer only (depth_idx=0)
    monthly_freq_result = get_monthly_extreme_data(bool_ds, year, month)
    if monthly_freq_result is None:
        print(f"ERROR: Could not calculate MHW frequency for {year}-{month:02d}")
        return None, None
    
    monthly_freq, time_indices = monthly_freq_result
    
    # Create mask from surface layer
    print(f"Creating mask for MHW frequency between {min_threshold} and {MHW_THRESHOLD_MAX} at surface layer...")
    mask = create_mhw_contour_mask(monthly_freq, min_threshold, MHW_THRESHOLD_MAX)
    
    # Calculate statistics for the variable with the mask
    print(f"Calculating {variable_name} statistics with surface MHW mask...")
    stats = calculate_masked_variable_stats(variable_ds, variable_config, mask)
    if stats is None:
        print(f"ERROR: Could not calculate statistics for {variable_name} in {year}-{month:02d}")
        return None, None
    
    # Load climatology data for the same month
    clim_datasets = load_all_climatology_data(month)
    if not clim_datasets:
        print(f"ERROR: Could not load sufficient data for climatology for {month:02d}")
        return stats, None
    
    # Calculate climatology statistics using the same mask
    print(f"Calculating climatology with the same mask...")
    clim_stats = calculate_climatology_stats(clim_datasets, variable_config, mask)
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
            if i > 0:
                dates.append(dates[-1] + pd.Timedelta(days=1))
            else:
                dates.append(pd.Timestamp(f"{year}-{month:02d}-01"))
    
    # Add dates to stats
    stats['dates'] = dates
    
    return stats, clim_stats

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
    
    print(f"\nProcessing {variable_title} timeseries (May-Oct only)...")
    
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
            
            # Format x-axis to focus on May-October period
            ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=ANALYSIS_MONTHS))
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
            
            # Set x-axis limits to include only May-Oct range
            if len(PRE_BLOB_YEARS) > 0:
                x_min = pd.Timestamp(f"{min(PRE_BLOB_YEARS)}-04-01")
            else:
                x_min = pd.Timestamp(f"{min(BLOB_YEARS)}-04-01")
                
            if len(POST_BLOB_YEARS) > 0:
                x_max = pd.Timestamp(f"{max(POST_BLOB_YEARS)}-11-30")
            else:
                x_max = pd.Timestamp(f"{max(BLOB_YEARS)}-11-30")
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
        
        # Format x-axis to focus on May-October period
        ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=ANALYSIS_MONTHS))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        
        # Set x-axis limits to include only May-Oct range
        if len(PRE_BLOB_YEARS) > 0:
            x_min = pd.Timestamp(f"{min(PRE_BLOB_YEARS)}-05-01")
        else:
            x_min = pd.Timestamp(f"{min(BLOB_YEARS)}-05-01")
            
        if len(POST_BLOB_YEARS) > 0:
            x_max = pd.Timestamp(f"{max(POST_BLOB_YEARS)}-10-31")
        else:
            x_max = pd.Timestamp(f"{max(BLOB_YEARS)}-10-31")
        ax1.set_xlim([x_min, x_max])
        ax2.set_xlim([x_min, x_max])
        
        # Add overall title
        plt.suptitle(f"{variable_title} Analysis ({STUDY_REGION['name']}) - Threshold {min_threshold*100:.0f}%", 
                   fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure with threshold in filename
        output_file = os.path.join(OUTPUT_DIR, 
                                 f"{variable_name}_timeseries_may_oct_threshold_{int(min_threshold*100)}.png")
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

def main():
    """Main execution function."""
    print("-" * 80)
    print(f"COMPLETE TIMESERIES ANALYSIS (2011-2021) WITH SURFACE LAYER MASK")
    print(f"Pre-Blob (2011-2013): Full Domain")
    print(f"Blob (2014-2016): Marine Heatwave Areas (surface mask)")
    print(f"Post-Blob (2017-2021): Full Domain")
    print(f"MONTHS: May (005) through October (010) only")
    print("-" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load boolean dataset with MHW data
    bool_data = load_boolean_data()
    bool_ds = bool_data[0] if bool_data else None
    
    if bool_ds is None:
        print("WARNING: Boolean dataset could not be loaded. Will process without MHW mask.")
    
    # Process each variable
    for variable_config in VARIABLES:
        process_complete_timeseries(variable_config, bool_ds)
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()