"""
Plot anomalies 2014-2016 of selected variables.

Author: Beni Blaser  
This script was developed with assistance from Claude Sonnet 3.7 (via Copilot), which supported code creation, debugging, and documentation.
"""

#%%
#--------- IMPORT REQUIRED MODULES ---------#
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean as cmo
import os
import glob
import re
import gc
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
import calendar

# Create a custom colormap using only the blue portion of diff
def create_blue_diff_cmap():
    # Get the colormap
    diff_cmap = cmo.cm.diff
    
    # Extract only the first half (blue portion) of the diff colormap
    # The 0.0 to 0.5 range contains the blue gradient part of the diverging colormap
    blue_colors = diff_cmap(np.linspace(0.5, 0.0, 256))
    
    # Create a new colormap with just these colors
    blue_diff = LinearSegmentedColormap.from_list('blue_diff', blue_colors)
    
    return blue_diff

# Add this function to create discrete colormaps with exactly 4 levels
def create_discrete_colormap(cmap_name, levels=8):
    """
    Creates a discrete colormap with exactly the specified number of levels.
    
    Parameters:
    -----------
    cmap_name : matplotlib colormap or string
        The base colormap to discretize
    levels : int
        Number of discrete levels to create
    
    Returns:
    --------
    cmap_discrete : LinearSegmentedColormap
        Discretized colormap
    norm : BoundaryNorm
        Normalization that maps values to discrete colors
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import numpy as np
    
    # Get the continuous colormap
    if isinstance(cmap_name, str):
        cmap = plt.cm.get_cmap(cmap_name)
    else:
        cmap = cmap_name
        
    # Extract colors for each level
    colors = cmap(np.linspace(0, 1, levels))
    
    # Create a new colormap from these colors
    cmap_discrete = ListedColormap(colors)
    
    # Create boundary norm based on the number of levels
    # This will map value ranges to the discrete colors
    bounds = np.linspace(0, 1, levels + 1)
    norm = BoundaryNorm(bounds, cmap_discrete.N)
    
    return cmap_discrete, norm

# Add this function to create discrete colormaps with 5 positive and 5 negative levels
def create_discrete_anomaly_colormap(cmap_name, vmin, vmax, levels=10):
    """
    Creates a discrete colormap with specified positive and negative levels.

    Parameters:
    -----------
    cmap_name : matplotlib colormap or string
        The base colormap to discretize
    vmin : float
        Minimum value for the colormap
    vmax : float
        Maximum value for the colormap
    levels : int
        Total number of discrete levels (default is 10: 5 positive, 5 negative)

    Returns:
    --------
    cmap_discrete : ListedColormap
        Discretized colormap
    norm : BoundaryNorm
        Normalization that maps values to discrete colors
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure levels is even to have equal positive and negative levels
    if levels % 2 != 0:
        raise ValueError("Levels must be an even number to have equal positive and negative levels.")

    # Get the continuous colormap
    if isinstance(cmap_name, str):
        cmap = plt.cm.get_cmap(cmap_name)
    else:
        cmap = cmap_name

    # Create boundaries for discrete levels
    bounds = np.linspace(vmin, vmax, levels + 1)
    colors = cmap(np.linspace(0, 1, levels))

    # Create a new colormap from these colors
    cmap_discrete = ListedColormap(colors)

    # Create a normalization object
    norm = BoundaryNorm(bounds, cmap_discrete.N)

    return cmap_discrete, norm

def create_symmetric_discrete_colormap(cmap_name, levels=9):
    """
    Creates a discrete symmetric diverging colormap with the middle level at zero.
    
    Parameters:
    -----------
    cmap_name : matplotlib colormap or string
        The base colormap to discretize
    levels : int
        Total number of discrete levels (must be odd for center=0)
        
    Returns:
    --------
    cmap_discrete : ListedColormap
        Discretized colormap
    norm : BoundaryNorm
        Normalization that maps values to discrete colors
    bounds : ndarray
        Boundary values for the colormap levels
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Ensure levels is odd to have a middle level at zero
    if levels % 2 == 0:
        levels += 1
    
    # Get the continuous colormap
    if isinstance(cmap_name, str):
        cmap = plt.cm.get_cmap(cmap_name)
    else:
        cmap = cmap_name
    
    # Extract colors for each level (evenly spaced)
    colors = cmap(np.linspace(0, 1, levels))
    
    # Create discrete colormap
    cmap_discrete = ListedColormap(colors)
    
    # Note: bounds will be set in the plotting function based on actual data
    
    return cmap_discrete

#--------- CONFIGURATION ---------#
DATA_DIR = '/nfs/sea/work/bblaser/z_avg_meanpool_domain/'
CLIM_DIR = '/nfs/sea/work/bblaser/monthly_clim/'
OUTPUT_DIR = '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events'
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Variables to process
VARIABLES = ['temp', 'TOT_PROD', 'POC_FLUX_IN']

# Years for anomaly calculation
ANOMALY_YEARS = [2014, 2015, 2016]
REFERENCE_YEARS = list(range(2011, 2021 + 1))
REFERENCE_YEARS = [year for year in REFERENCE_YEARS if year not in ANOMALY_YEARS]

# Configure which months to process (1-12)
# 1=Jan, 2=Feb, etc. Add or remove months as needed
MONTHS_TO_PROCESS = [5, 6, 7, 8, 9, 10]  # May through October

# Depth configurations
DEPTH_LEVELS_50M = [0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50]
POC_FLUX_DEPTH = -100  # Use 100m depth for POC_FLUX_IN

# Plot configurations
COLORMAPS = {
    'temp': cmo.cm.amp,
    'TOT_PROD': cmo.cm.tempo,
    'POC_FLUX_IN': create_blue_diff_cmap()  # Use our custom blue colormap instead of ice_r
}

# Value ranges for colormaps
VLIMS = {
    'temp': {'clim': [5, 25], 'anom': [-1.2, 1.2]},
    'TOT_PROD': {'clim': [0, 0.6], 'anom': [-0.0000025 * 31536.0, 0.0000025 * 31536.0]},
    'POC_FLUX_IN': {'clim': [0, 6.0], 'anom': [-0.00003 * 31536.0, 0.00003 * 31536.0]}
}

# Plot titles and labels
TITLES = {
    'temp': 'Temperature',
    'TOT_PROD': 'NPP',
    'POC_FLUX_IN': 'POC Flux'
}

# Define conversion factors for the units
UNIT_CONVERSION = {
    'temp': 1.0,  # No conversion needed
    'TOT_PROD': 31536.0,  # Convert mmol/m^3/s to mol/m^3/year
    'POC_FLUX_IN': 31536.0  # Convert mmolC/m^2/s to molC/m^2/year
}

# The VLIMS will be calculated dynamically based on percentiles
# We'll keep these as fallback values
VLIMS = {
    'temp': {'clim': [5, 25], 'anom': [-1.2, 1.2]},
    'TOT_PROD': {'clim': [0, 0.6], 'anom': [-0.0000025 * 31536.0, 0.0000025 * 31536.0]},
    'POC_FLUX_IN': {'clim': [0, 6.0], 'anom': [-0.00003 * 31536.0, 0.00003 * 31536.0]}
}

# Add this near the other configuration variables
TITLES_UNITS = {
    'temp': '°C',
    'TOT_PROD': 'mol m⁻³ yr⁻¹',
    'POC_FLUX_IN': 'mol C m⁻² yr⁻¹'
}

# Define seasons by month numbers
SEASONS = {
    'Winter': [10, 11, 12, 1, 2, 3],  # Oct-Mar
    'Summer': [4, 5, 6, 7, 8, 9]      # Apr-Sept
}

# Seasons to process
SEASONS_TO_PROCESS = ['Winter', 'Summer']

#%%
#--------- HELPER FUNCTIONS ---------#
def get_study_region():
    """Define the study region bounds based on plot extent."""
    return {
        'lon_min': 205,
        'lon_max': 245,
        'lat_min': 20,
        'lat_max': 62
    }

def get_depth_levels(variable):
    """Get appropriate depth levels for each variable."""
    if variable == 'POC_FLUX_IN':
        return [POC_FLUX_DEPTH]
    return DEPTH_LEVELS_50M

def slice_to_region(dataset, variable=None):
    """
    Efficient region slicing for both Dataset and DataArray objects.
    Uses xarray's built-in selection methods for coordinates/data variables.
    """
    if dataset is None:
        return None
        
    region = get_study_region()
    
    # Identify coordinate variables
    lon_var = None
    lat_var = None
    
    # Check for longitude in data variables first, then coordinates
    for lon_name in ['lon_rho', 'lon']:
        if lon_name in dataset.data_vars:
            lon_var = lon_name
            break
        elif lon_name in dataset.coords:
            lon_var = lon_name
            break
            
    # Check for latitude in data variables first, then coordinates
    for lat_name in ['lat_rho', 'lat']:
        if lat_name in dataset.data_vars:
            lat_var = lat_name
            break
        elif lat_name in dataset.coords:
            lat_var = lat_name
            break
    
    # If we found both coordinates, promote them to coords and slice
    if lon_var and lat_var:
        # make lon/lat real coords
        dataset = dataset.assign_coords(lon=dataset[lon_var], lat=dataset[lat_var])
        lon_var, lat_var = 'lon', 'lat'

        print(f"  Slicing data to region: lon=[{region['lon_min']}-{region['lon_max']}], lat=[{region['lat_min']}-{region['lat_max']}]")
        
        try:
            # Get the lon/lat arrays for efficient indexing
            lon_array = dataset[lon_var].values
            lat_array = dataset[lat_var].values
            
            # Create a mask for the region
            if lat_array.ndim == 2 and lon_array.ndim == 2:
                # 2D coordinate arrays (common in curvilinear grids)
                lon_mask = (lon_array >= region['lon_min']) & (lon_array <= region['lon_max'])
                lat_mask = (lat_array >= region['lat_min']) & (lat_array <= region['lat_max'])
                combined_mask = lon_mask & lat_mask
                
                # Get indices where both conditions are satisfied
                valid_indices = np.where(combined_mask)
                
                if len(valid_indices[0]) > 0:
                    min_eta, max_eta = valid_indices[0].min(), valid_indices[0].max()
                    min_xi, max_xi = valid_indices[1].min(), valid_indices[1].max()
                    
                    # Use isel for direct indexing
                    return dataset.isel(eta_rho=slice(min_eta, max_eta+1), 
                                       xi_rho=slice(min_xi, max_xi+1))
            else:
                # Use where method for 1D coordinate arrays
                return dataset.where(
                    (dataset[lon_var] >= region['lon_min']) & 
                    (dataset[lon_var] <= region['lon_max']) &
                    (dataset[lat_var] >= region['lat_min']) & 
                    (dataset[lat_var] <= region['lat_max']),
                    drop=True
                )
                
        except Exception as e:
            print(f"  Warning: Region slicing failed with error: {e}")
            # Return the original dataset if slicing fails
            return dataset
    
    print(f"  Warning: Could not identify both lat/lon coordinates for slicing.")
    return dataset

def load_climatology_file(month_str, variable=None):
    """Load a monthly climatology file directly."""
    clim_file = os.path.join(CLIM_DIR, f"climatology_month_{month_str}_2011-2021.nc")
    
    if not os.path.exists(clim_file):
        print(f"Warning: Climatology file not found: {clim_file}")
        return None
        
    try:
        ds = xr.open_dataset(clim_file)
        
        # Immediately slice to region to reduce memory usage
        ds = slice_to_region(ds, variable)
        
        return ds
    except Exception as e:
        print(f"Error opening climatology file {os.path.basename(clim_file)}: {e}")
        return None

def load_data_file(year, month_str, variable=None):
    """Load a data file by year and month string."""
    file_pattern = os.path.join(DATA_DIR, f"z_avg_{year}_{month_str}_37zlevs_full_1x1meanpool_downsampling.nc")
    matching_files = glob.glob(file_pattern)
    
    if not matching_files:
        print(f"No files found for year {year}, month {month_str}")
        return None
        
    try:
        ds = xr.open_dataset(matching_files[0])
        
        # Immediately slice to region to reduce memory usage
        ds = slice_to_region(ds, variable)
        
        return ds
    except Exception as e:
        print(f"Error opening file for {year}/{month_str}: {e}")
        return None

def compute_depth_average(ds, variable, depth_levels):
    """Calculate average over specific depth levels for a dataset."""
    if variable not in ds:
        print(f"Error: Variable {variable} not found in dataset")
        return None
    
    # Identify the depth dimension
    depth_dim = None
    for dim in ds[variable].dims:
        if dim not in ['time', 'xi_rho', 'eta_rho']:
            depth_dim = dim
            break
    
    if not depth_dim:
        print("Warning: Could not identify depth dimension")
        return ds[variable]
    
    # Create list to store depth slices
    depth_slices = []
    
    # Handle depth selection
    if hasattr(ds[depth_dim], 'values') and len(ds[depth_dim].values) > 0:
        # Select by actual depth values
        for depth in depth_levels:
            try:
                slice_data = ds[variable].sel({depth_dim: depth}, method='nearest')
                depth_slices.append(slice_data)
            except Exception:
                print(f"Error selecting depth {depth}, falling back to indices")
                # Fall back to indices if selection fails
                for i in range(min(len(depth_levels), len(ds[depth_dim]))):
                    depth_slices.append(ds[variable].isel({depth_dim: i}))
                break
    else:
        # Use indices directly
        for i in range(min(len(depth_levels), len(ds[depth_dim]))):
            depth_slices.append(ds[variable].isel({depth_dim: i}))
    
    # If we have depth slices, average them
    if depth_slices:
        # Create a synthetic dimension for depth slices
        stacked = xr.concat(depth_slices, dim='z_levels')
        # Return the mean across depths
        return stacked.mean(dim='z_levels', skipna=True)
    else:
        print(f"Warning: No depth slices created for {variable}")
        return ds[variable]

#--------- MAIN LOADING FUNCTIONS ---------#
def load_climatology_data(variable, monthly=False):
    """
    Load climatology data for a given variable.
    Returns monthly climatologies if monthly=True, otherwise annual mean.
    """
    print(f"Loading climatology data for {variable}...")
    
    # Storage for monthly data
    monthly_data = {}
    
    # Loop through all months (1-12)
    for month in range(5, 11):
        month_str = f"{month:03d}"
        
        # Load climatology file
        ds = load_climatology_file(month_str, variable)
        
        if ds is None or variable not in ds:
            print(f"  Warning: Could not load {variable} for month {month}")
            continue
            
        # Get depth levels appropriate for this variable
        depth_levels = get_depth_levels(variable)
        
        # Calculate depth average
        depth_avg = compute_depth_average(ds, variable, depth_levels)
        
        # Immediately slice to region limits
        region = get_study_region()
        if depth_avg is not None:
            # Make sure lon/lat are available as coordinates
            if 'lon' in depth_avg.coords and 'lat' in depth_avg.coords:
                depth_avg = depth_avg.where(
                    (depth_avg.lon >= region['lon_min']) & 
                    (depth_avg.lon <= region['lon_max']) &
                    (depth_avg.lat >= region['lat_min']) & 
                    (depth_avg.lat <= region['lat_max']),
                    drop=True
                )
            monthly_data[month] = depth_avg
            
        # Close dataset to free memory
        ds.close()
        del ds
        gc.collect()
    
    if not monthly_data:
        print(f"Error: Could not load any climatology data for {variable}")
        return None
    
    if monthly:
        # Return a dictionary of monthly data
        print(f"Successfully loaded monthly climatologies for {variable}")
        return monthly_data
    else:
        # Calculate annual mean
        monthly_array = xr.concat(list(monthly_data.values()), dim='month')
        annual_mean = monthly_array.mean(dim='month')
        print(f"Successfully loaded annual climatology for {variable}")
        return annual_mean

def load_anomaly_data(variable, years_tuple):
    """
    Load anomaly data for specified years.
    Calculates anomalies against the monthly climatology.
    """
    print(f"Loading anomaly data for {variable} for years {list(years_tuple)}...")
    
    # Convert tuple to list for iteration
    years = list(years_tuple)
    
    # Load monthly climatologies
    monthly_clim = load_climatology_data(variable, monthly=True)
    if not monthly_clim:
        return None
    
    # Storage for anomaly data by year and month
    anomaly_data = {}
    
    for year in years:
        anomaly_data[year] = {}
        
        for month in range(5, 11):
            month_str = f"{month:03d}"
            
            # Load actual data file
            ds = load_data_file(year, month_str, variable)
            
            if ds is None or variable not in ds:
                continue
            
            # Get depth levels appropriate for this variable
            depth_levels = get_depth_levels(variable)
            
            # Calculate depth average for actual data
            month_depth_avg = compute_depth_average(ds, variable, depth_levels)
            
            if month_depth_avg is not None and month in monthly_clim:
                # If file has daily data, calculate monthly mean
                if 'time' in month_depth_avg.dims:
                    month_mean = month_depth_avg.mean(dim='time')
                else:
                    month_mean = month_depth_avg
                
                # Calculate anomaly against climatology
                anomaly = month_mean - monthly_clim[month]
                anomaly_data[year][month] = anomaly
            
            # Close dataset to free memory
            ds.close()
            del ds
            gc.collect()
    
    # Combine all anomalies into one dataset
    all_anomalies = []
    for year in anomaly_data:
        for month in anomaly_data[year]:
            all_anomalies.append(anomaly_data[year][month])
    
    if not all_anomalies:
        print(f"Error: No anomalies calculated for {variable}")
        return None
        
    # Combine all anomalies and calculate mean
    anomaly_mean = xr.concat(all_anomalies, dim='time').mean(dim='time')
    print(f"Successfully calculated anomaly data for {variable} using monthly differences")
    
    return anomaly_mean

def load_anomaly_data_seasonal(variable, years_tuple, season):
    """
    Load anomaly data for specified years and season.
    Calculates anomalies against the monthly climatology.
    
    Parameters:
    - variable: Variable to process
    - years_tuple: Years to include in anomaly calculation
    - season: Season name to filter data ('Winter' or 'Summer')
    """
    print(f"Loading anomaly data for {variable} during {season} for years {list(years_tuple)}...")
    
    # Convert tuple to list for iteration
    years = list(years_tuple)
    
    # Load monthly climatologies
    monthly_clim = load_climatology_data(variable, monthly=True)
    if not monthly_clim:
        return None
    
    # Get months for the specified season
    season_months = SEASONS.get(season, [])
    if not season_months:
        print(f"Error: Invalid season '{season}'")
        return None
        
    print(f"  Including months: {season_months}")
    
    # Storage for anomaly data by year and month
    anomaly_data = {}
    
    for year in years:
        anomaly_data[year] = {}
        
        for month in range(5, 11):
            # Skip if month is not in the specified season
            if month not in season_months:
                continue
                
            month_str = f"{month:03d}"
            
            # Special handling for winter season spanning across years
            if season == 'Winter' and month in [1, 2, 3]:
                # For January-March, check if we have data for year+1
                if year + 1 not in years:
                    continue
                    
                # Load actual data file from the next year for Jan-Mar
                ds = load_data_file(year + 1, month_str, variable)
            else:
                # Load actual data file from the current year
                ds = load_data_file(year, month_str, variable)
            
            if ds is None or variable not in ds:
                continue
            
            # Get depth levels appropriate for this variable
            depth_levels = get_depth_levels(variable)
            
            # Calculate depth average for actual data
            month_depth_avg = compute_depth_average(ds, variable, depth_levels)
            
            if month_depth_avg is not None and month in monthly_clim:
                # If file has daily data, calculate monthly mean
                if 'time' in month_depth_avg.dims:
                    month_mean = month_depth_avg.mean(dim='time')
                else:
                    month_mean = month_depth_avg
                
                # Calculate anomaly against climatology
                anomaly = month_mean - monthly_clim[month]
                
                # Store in the appropriate year bucket
                if season == 'Winter' and month in [1, 2, 3]:
                    if year + 1 not in anomaly_data:
                        anomaly_data[year + 1] = {}
                    anomaly_data[year + 1][month] = anomaly
                else:
                    anomaly_data[year][month] = anomaly
            
            # Close dataset to free memory
            ds.close()
            del ds
            gc.collect()
    
    # Combine all anomalies into one dataset
    all_anomalies = []
    for year in anomaly_data:
        for month in anomaly_data[year]:
            all_anomalies.append(anomaly_data[year][month])
    
    if not all_anomalies:
        print(f"Error: No anomalies calculated for {variable} during {season}")
        return None
        
    # Combine all anomalies and calculate mean
    anomaly_mean = xr.concat(all_anomalies, dim='time').mean(dim='time')
    print(f"Successfully calculated seasonal ({season}) anomaly data for {variable} using {len(all_anomalies)} months")
    
    return anomaly_mean

#%%
#--------- PLOTTING FUNCTIONS ---------#
def plot_climatology_vs_anomaly(variable):
    """Create a side-by-side plot of climatology and anomalies for a given variable."""
    print(f"Plotting climatology and anomalies for {variable}...")
    
    # Load data
    print("  Loading climatology...")
    clim_data = load_climatology_data(variable)
    
    print("  Loading anomaly data...")
    anom_data = load_anomaly_data(variable, tuple(ANOMALY_YEARS))
    
    if clim_data is None or anom_data is None:
        print(f"Error: Could not load data for {variable}")
        return
    
    # Apply unit conversion
    conversion_factor = UNIT_CONVERSION[variable]
    if conversion_factor != 1.0:
        print(f"  Applying unit conversion factor: {conversion_factor}")
        clim_data = clim_data * conversion_factor
        anom_data = anom_data * conversion_factor
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300, 
                            gridspec_kw={'width_ratios': [1, 1]})
    
    # Get the colormap and value ranges for this variable
    cmap = COLORMAPS.get(variable, 'viridis')
    vmin_clim, vmax_clim = VLIMS.get(variable, {}).get('clim', [None, None])
    vmin_anom, vmax_anom = VLIMS.get(variable, {}).get('anom', [None, None])
    # choose anomaly colormap: balance for temp, tarn for TOT_PROD & POC_FLUX_IN
    if variable == 'temp':
        anom_cmap = cmo.cm.balance
    elif variable =='TOT_PROD':  # You need to define what condition to check here
        anom_cmap = cmo.cm.tarn
    else:
        anom_cmap = cmo.cm.diff_r
        
    # Create discrete colormap for anomaly with 5 positive and 5 negative levels
    discrete_cmap, discrete_norm = create_discrete_anomaly_colormap(
        anom_cmap, vmin_anom, vmax_anom, levels=10
    )
    
    # Get coordinates for plotting
    lon_var = None
    lat_var = None
    
    # Check for coordinates in the DataArray - corrected for DataArray objects
    if isinstance(clim_data, xr.DataArray):
        # For DataArray objects, check only coords
        for coord_name in ['lon_rho', 'lon']:
            if coord_name in clim_data.coords:
                lon_var = coord_name
                break
        
        for coord_name in ['lat_rho', 'lat']:
            if coord_name in clim_data.coords:
                lat_var = coord_name
                break
    else:
        # For Dataset objects, check both coords and data_vars
        for coord_name in ['lon_rho', 'lon']:
            if coord_name in clim_data.coords or coord_name in clim_data.data_vars:
                lon_var = coord_name
                break
        
        for coord_name in ['lat_rho', 'lat']:
            if coord_name in clim_data.coords or coord_name in clim_data.data_vars:
                lat_var = coord_name
                break
    
    # Compute data arrays (force computation)
    print("  Computing data arrays...")
    clim_values = clim_data.compute().values
    anom_values = anom_data.compute().values
    
    # Plot using proper coordinates if found
    if lon_var and lat_var:
        print(f"  Using coordinates: {lon_var}, {lat_var}")
        lon_data = clim_data[lon_var].compute()
        lat_data = clim_data[lat_var].compute()
        
        # Handle 2D coordinate arrays
        if lon_data.ndim == 2 and lat_data.ndim == 2:
            print("  Using 2D coordinate arrays for plotting")
            # Plot climatology
            print("  Creating climatology plot...")
            pcm1 = axes[0].pcolormesh(lon_data.values, lat_data.values, clim_values, 
                                   cmap=cmap, vmin=vmin_clim, vmax=vmax_clim)
            
            # Plot anomaly
            print("  Creating anomaly plot...")
            pcm2 = axes[1].pcolormesh(lon_data.values, lat_data.values, anom_values, 
                                   cmap=discrete_cmap, norm=discrete_norm)
        else:
            print("  Using 1D coordinate arrays for plotting")
            # For 1D coordinate arrays
            pcm1 = axes[0].pcolormesh(lon_data.values, lat_data.values, clim_values, 
                                  cmap=cmap, vmin=vmin_clim, vmax=vmax_clim)
            pcm2 = axes[1].pcolormesh(lon_data.values, lat_data.values, anom_values, 
                                  cmap=discrete_cmap, norm=discrete_norm)
    else:
        print("Warning: Coordinates not found, using index-based plotting")
        pcm1 = axes[0].pcolormesh(clim_values, cmap=cmap, vmin=vmin_clim, vmax=vmax_clim)
        pcm2 = axes[1].pcolormesh(anom_values, cmap=discrete_cmap, norm=discrete_norm)
    
    # Add colorbar for climatology
    cbar1 = fig.colorbar(pcm1, ax=axes[0])
    cbar1.set_label(f"{TITLES.get(variable, variable)} - Baseline Period")
    
    # Add colorbar for anomaly
    cbar2 = fig.colorbar(pcm2, ax=axes[1])
    cbar2.set_label(f"{TITLES.get(variable, variable)} Anomaly with respect to Baseline Period")
    
    # Set common plot properties
    for i, title in enumerate(["Baseline Period (2010-2021)", 
                             f"Anomaly ({', '.join(map(str, ANOMALY_YEARS))}) vs Reference"]):
        axes[i].set_title(title)
        axes[i].set_xlabel("Longitude (°E)")
        if i == 0:
            axes[i].set_ylabel("Latitude (°N)")
        
        # Set plot extent if coordinates are available
        if lon_var and lat_var:
            region = get_study_region()
            axes[i].set_xlim(region['lon_min'], region['lon_max'])
            axes[i].set_ylim(region['lat_min'], region['lat_max'])
            
        # Make each subplot square
        axes[i].set_aspect(1)
        
        # Add grid lines
        axes[i].grid(True, linestyle='--', alpha=0.3)
    
    # Add overall title
    plt.suptitle(f"{TITLES.get(variable, variable)} - Baseline Period vs. Anomaly", fontsize=16)
    
    # Save the figure
    output_file = os.path.join(OUTPUT_DIR, f"{variable}_climatology_vs_anomaly.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()
    
    # Clear memory
    del clim_data, anom_data, clim_values, anom_values
    gc.collect()
    
#%% 
#--------- COMBINED PLOTTING FUNCTION ---------#
def plot_all_combined():
    """Plot all variables in a 2×3 grid with climatologies on top row and anomalies on bottom row."""
    n = len(VARIABLES)
    print("\nCreating combined plot with data-driven limits and discrete colormap for anomalies...")
    
    # Increase figure size and adjust aspect ratio for better fit
    fig = plt.figure(figsize=(18, 12), dpi=300)
    
    # Reduce spacing between subplots significantly
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1], width_ratios=[1, 1, 1], 
                         hspace=0, wspace=0.4)
    
    # Get the study region bounds
    region = get_study_region()
    
    # Convert longitudes to -180 to 180 format for Cartopy
    lon_min_180 = region['lon_min'] - 360 if region['lon_min'] > 180 else region['lon_min']
    lon_max_180 = region['lon_max'] - 360 if region['lon_max'] > 180 else region['lon_max']
    
    for i, variable in enumerate(VARIABLES):
        print(f"  Processing {variable}...")
        
        # Create axes with proper projection using the GridSpec
        axc = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree(central_longitude=0))
        axa = fig.add_subplot(gs[1, i], projection=ccrs.PlateCarree(central_longitude=0))
        
        # Get depth level information and create proper depth label
        depth_levels = get_depth_levels(variable)
        if variable == 'POC_FLUX_IN':
            # For POC_FLUX_IN, use "at 100m" label
            depth_label = f"at {abs(depth_levels[0])}m"
        else:
            # For other variables with depth range, use "0m-50m" format
            depth_label = f"0m-{abs(min(depth_levels))}m mean"
        
        # Set the map extent using the converted coordinates
        axc.set_extent([lon_min_180, lon_max_180, 
                        region['lat_min'], region['lat_max']], 
                       crs=ccrs.PlateCarree())
        axa.set_extent([lon_min_180, lon_max_180, 
                        region['lat_min'], region['lat_max']], 
                       crs=ccrs.PlateCarree())
        
        # Add coastlines and land features
        axc.coastlines(resolution='50m', color='black', linewidth=0.5)
        axa.coastlines(resolution='50m', color='black', linewidth=0.5)
        axc.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        axa.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        
        # Add gridlines with proper configuration for tick display
        gl1 = axc.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                          alpha=0.5, linestyle='--')
        gl2 = axa.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                          alpha=0.5, linestyle='--')
        
        # Configure grid lines with explicit tick locations
        gl1.xlocator = mticker.FixedLocator(range(-160, -100, 10))
        gl1.ylocator = mticker.FixedLocator(range(20, 70, 10))
        gl2.xlocator = mticker.FixedLocator(range(-160, -100, 10))
        gl2.ylocator = mticker.FixedLocator(range(20, 70, 10))
        
        # Turn off right labels (and top labels for cleaner look)
        gl1.right_labels = False
        gl1.top_labels = False
        gl2.right_labels = False 
        gl2.top_labels = False
        
        # Ensure label styles are set to make ticks visible
        gl1.xlabel_style = {'size': 12, 'color': 'black'}
        gl1.ylabel_style = {'size': 12, 'color': 'black'}
        gl2.xlabel_style = {'size': 12, 'color': 'black'}
        gl2.ylabel_style = {'size': 12, 'color': 'black'}
        
        # Load data
        clim = load_climatology_data(variable)
        anom = load_anomaly_data(variable, tuple(ANOMALY_YEARS))
        if clim is None or anom is None:
            print(f"  Error: Could not load data for {variable}")
            continue
        
        # Apply unit conversion
        conversion_factor = UNIT_CONVERSION[variable]
        if conversion_factor != 1.0:
            clim = clim * conversion_factor
            anom = anom * conversion_factor
        
        # Get climatology colormap limits from VLIMS dictionary
        vmin_c, vmax_c = VLIMS[variable]['clim']
        
        # Calculate anomaly limits from the 99th percentile of the data
        abs_anom = np.abs(anom.values)
        # Remove NaN values before calculating percentile
        abs_anom = abs_anom[~np.isnan(abs_anom)]
        
        # Calculate 99th percentile value
        if len(abs_anom) > 0:
            vmax_a = np.percentile(abs_anom, 99)
            print(f"  99th percentile for {variable} anomaly: {vmax_a:.6f}")
        else:
            # Fallback to the predefined limits if no valid data
            vmax_a = VLIMS[variable]['anom'][1]
            print(f"  Using default vmax for {variable} anomaly: {vmax_a:.6f}")
        
        # Set symmetric limits
        vmin_a = -vmax_a
        
        # Get coordinates
        lon = clim.coords.get('lon', None)
        lat = clim.coords.get('lat', None)
        use_2d = lon is not None and lon.ndim == 2
        
        # Get colormaps
        cmap_base = COLORMAPS[variable]
        if variable == 'temp':
            anom_cmap_base = cmo.cm.balance
        elif variable == 'TOT_PROD':
            anom_cmap_base = cmo.cm.tarn
        else:
            anom_cmap_base = cmo.cm.diff_r
            
        # Create discrete colormap for climatology with 9 levels
        levels = 9
        clim_colors = cmap_base(np.linspace(0, 1, levels))
        discrete_clim_cmap = ListedColormap(clim_colors)
        clim_bounds = np.linspace(vmin_c, vmax_c, levels + 1)
        clim_norm = BoundaryNorm(clim_bounds, discrete_clim_cmap.N)
            
        # Create discrete colormap for anomalies
        discrete_anom_cmap = create_symmetric_discrete_colormap(anom_cmap_base, levels=9)
        
        # Create boundary norm based on the calculated limits
        anom_bounds = np.linspace(vmin_a, vmax_a, levels + 1)
        anom_norm = BoundaryNorm(anom_bounds, discrete_anom_cmap.N)
        
        # Compute data arrays
        clim_vals = clim.compute().values
        anom_vals = anom.compute().values
        
        # Plot with proper transform
        if use_2d:
            # When using 2D coordinates, ensure they're in proper format for plotting
            plot_lon = lon.values - 360 if np.any(lon.values > 180) else lon.values
            pc1 = axc.pcolormesh(plot_lon, lat, clim_vals, cmap=discrete_clim_cmap, 
                              norm=clim_norm, transform=ccrs.PlateCarree())
            pc2 = axa.pcolormesh(plot_lon, lat, anom_vals, cmap=discrete_anom_cmap, 
                              norm=anom_norm, transform=ccrs.PlateCarree())
        else:
            # For 1D coordinate arrays or direct plotting
            pc1 = axc.pcolormesh(clim_vals, cmap=discrete_clim_cmap, norm=clim_norm,
                             transform=ccrs.PlateCarree())
            pc2 = axa.pcolormesh(anom_vals, cmap=discrete_anom_cmap, norm=anom_norm,
                             transform=ccrs.PlateCarree())
        
        # Add titles
        axc.set_title(f"{TITLES[variable]} – Baseline Period (2011-2021)\n({depth_label})")
        axa.set_title(f"{TITLES[variable]} Anomaly (2014-2016 vs Baseline Period)\n({depth_label})")
        
        # Create proper colorbars with consistent height
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # For climatology (now discrete with 9 levels)
        divider1 = make_axes_locatable(axc)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
        
        # Use bounds as tick positions for climatology colorbar
        cbar1 = fig.colorbar(pc1, cax=cax1, ticks=clim_bounds)
        cbar1.set_label(f"{TITLES_UNITS[variable]}", size=12)  # Increase size to 14 or your preferred value
        
        # Format climatology tick labels with appropriate precision
        if variable == 'temp':
            cbar1.ax.set_yticklabels([f"{t:.1f}" for t in clim_bounds])
        elif variable == 'TOT_PROD':
            cbar1.ax.set_yticklabels([f"{t:.2f}" for t in clim_bounds])
        else:
            cbar1.ax.set_yticklabels([f"{t:.2f}" for t in clim_bounds])
        
        # For anomaly colorbar
        divider2 = make_axes_locatable(axa)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
        
        # Use bounds as tick positions for anomaly colorbar
        cbar2 = fig.colorbar(pc2, cax=cax2, ticks=anom_bounds)
        cbar2.set_label(f"Δ {TITLES_UNITS[variable]}", size=12)  # Increase size to 14 or your preferred value
        
        # Format anomaly tick labels with appropriate precision
        if variable == 'temp':
            cbar2.ax.set_yticklabels([f"{t:.1f}" for t in anom_bounds])
        elif variable == 'TOT_PROD':
            cbar2.ax.set_yticklabels([f"{t:.2f}" for t in anom_bounds])
        else:
            cbar2.ax.set_yticklabels([f"{t:.1f}" for t in anom_bounds])
        
        # Ensure ticks are visible by setting tick parameters
        cbar1.ax.tick_params(labelsize=12, width=1, length=5)
        cbar2.ax.tick_params(labelsize=12, width=1, length=5)
    
    # Add overall title with slightly reduced y position
    if MONTHS_TO_PROCESS:
        first_month = calendar.month_name[min(MONTHS_TO_PROCESS)]
        last_month = calendar.month_name[max(MONTHS_TO_PROCESS)]
        plt.suptitle(f'Northeast Pacific Ocean - Comparison of Baseline Periods and Anomalies (2011-2021 vs 2014-2016)\n{first_month}-{last_month}', 
                    fontsize=16, y=0.925)
    else:
        plt.suptitle('Northeast Pacific Ocean - Comparison of Baseline Periods and Anomalies (2011-2021 vs 2014-2016)', 
                    fontsize=16, y=0.925)
    
    # Save figure with tight layout to further reduce whitespace
    output_file = os.path.join(OUTPUT_DIR, 'combined_anomaly_analysis_9levels.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close(fig)

def plot_all_combined_seasonal():
    """Plot all variables with separate rows for each season."""
    n_vars = len(VARIABLES)
    n_seasons = len(SEASONS_TO_PROCESS)
    
    # Create a figure with rows for seasons*variables
    fig = plt.figure(figsize=(16, 9 * n_vars * n_seasons), dpi=300)
    
    # Get the study region bounds
    region = get_study_region()
    
    # Convert longitudes to -180 to 180 format for Cartopy
    lon_min_180 = region['lon_min'] - 360 if region['lon_min'] > 180 else region['lon_min']
    lon_max_180 = region['lon_max'] - 360 if region['lon_max'] > 180 else region['lon_max']
    
    subplot_idx = 1
    
    # First load all climatologies to save time
    climatologies = {}
    for variable in VARIABLES:
        climatologies[variable] = load_climatology_data(variable)
    
    # Loop through variables and seasons
    for var_idx, variable in enumerate(VARIABLES):
        clim = climatologies[variable]
        if clim is None:
            continue
            
        # Apply unit conversion
        conversion_factor = UNIT_CONVERSION[variable]
        if conversion_factor != 1.0:
            clim = clim * conversion_factor
            
        # Get coordinates
        lon = clim.coords.get('lon', None)
        lat = clim.coords.get('lat', None)
        use_2d = lon is not None and lon.ndim == 2
        
        for season_idx, season in enumerate(SEASONS_TO_PROCESS):
            # Create axes with proper projection
            axc = fig.add_subplot(n_vars * n_seasons, 2, subplot_idx, 
                                  projection=ccrs.PlateCarree(central_longitude=0))
            subplot_idx += 1
            axa = fig.add_subplot(n_vars * n_seasons, 2, subplot_idx, 
                                  projection=ccrs.PlateCarree(central_longitude=0))
            subplot_idx += 1
            
            # Get depth level information and create proper depth label
            depth_levels = get_depth_levels(variable)
            if variable == 'POC_FLUX_IN':
                depth_label = f"at {abs(depth_levels[0])}m"
            else:
                depth_label = f"0m-{abs(min(depth_levels))}m mean"
            
            # Set up the map
            for ax in [axc, axa]:
                ax.set_extent([lon_min_180, lon_max_180, 
                              region['lat_min'], region['lat_max']], 
                             crs=ccrs.PlateCarree())
                ax.coastlines(resolution='50m', color='black', linewidth=0.5)
                ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
                
                # Add gridlines manually
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                                alpha=0.5, linestyle='--', x_inline=False, y_inline=False)
                gl.xlocator = mticker.MultipleLocator(10)
                gl.ylocator = mticker.MultipleLocator(10)
                gl.top_labels = False
                
                # Configure labels for left/right panels
                if ax == axc:  # Left panel (climatology)
                    gl.right_labels = False
                else:  # Right panel (anomaly)
                    gl.left_labels = False
                    gl.right_labels = False
            
            # Load seasonal anomaly data
            anom = load_anomaly_data_seasonal(variable, tuple(ANOMALY_YEARS), season)
            if anom is None:
                continue
            
            # Apply unit conversion to anomaly
            if conversion_factor != 1.0:
                anom = anom * conversion_factor
            
            # Get colormap limits from VLIMS dictionary
            vmin_c, vmax_c = VLIMS[variable]['clim']
            vmin_a, vmax_a = VLIMS[variable]['anom']
            
            # Get colormaps
            cmap = COLORMAPS[variable]
            if variable == 'temp':
                anom_cmap_base = cmo.cm.balance
            elif variable =='TOT_PROD':
                anom_cmap_base = cmo.cm.tarn
            else:
                anom_cmap_base = cmo.cm.diff_r
            
            # Create discrete colormap for anomaly with exactly 4 levels
            from matplotlib.colors import BoundaryNorm
            discrete_cmap, norm, bounds = create_symmetric_discrete_colormap(
                anom_cmap_base, vmin_a, vmax_a, levels=9)
            
            # Compute data arrays
            clim_vals = clim.compute().values
            anom_vals = anom.compute().values
            
            # Plot with proper transform
            if use_2d:
                # When using 2D coordinates, ensure they're in proper format for plotting
                plot_lon = lon - 360 if np.any(lon > 180) else lon
                pc1 = axc.pcolormesh(plot_lon, lat, clim_vals, cmap=cmap, vmin=vmin_c, vmax=vmax_c, 
                                 transform=ccrs.PlateCarree())
                pc2 = axa.pcolormesh(plot_lon, lat, anom_vals, cmap=discrete_cmap, norm=norm,
                                 transform=ccrs.PlateCarree(), vmin=bounds[0], vmax=bounds[-1])
            else:
                # For 1D coordinate arrays or direct plotting
                pc1 = axc.pcolormesh(clim_vals, cmap=cmap, vmin=vmin_c, vmax=vmax_c,
                                  transform=ccrs.PlateCarree())
                pc2 = axa.pcolormesh(anom_vals, cmap=discrete_cmap, norm=norm,
                                  transform=ccrs.PlateCarree(), vmin=bounds[0], vmax=bounds[-1])
            
            # Add titles and labels
            axc.set_title(f"{TITLES[variable]} – Baseline Period (2011-2021)\n({depth_label})")
            
            # Season-specific title
            season_months_str = "Oct-Mar" if season == "Winter" else "Apr-Sep"
            axa.set_title(f"{TITLES[variable]} {season} ({season_months_str}) Anomaly\n"
                         f"2014-2016 vs Baseline Period ({depth_label})")
            
            # Add colorbars with units
            plt.colorbar(pc1, ax=axc, label=f"{TITLES[variable]} (°C)" if variable == 'temp' else 
                          f"{TITLES[variable]} ({TITLES_UNITS[variable]})")
            
            # For the colorbar, make sure to specify the bounds:
            cb2 = plt.colorbar(pc2, ax=axa, 
                        label=f"Deviation (°C)" if variable == 'temp' else f"Deviation ({TITLES_UNITS[variable]})",
                        ticks=bounds)
            
            # Format colorbar tick labels to be more readable
            cb2.ax.set_yticklabels([f"{x:.2g}" for x in bounds])
            
            # Replace the colorbar code section in plot_all_combined_seasonal():
            # Get custom ticks for this variable
            custom_ticks, tick_labels = get_custom_ticks_for_variable(
                variable, vmin_a, vmax_a)

            cb2 = plt.colorbar(pc2, ax=axa, 
                              label=f"Deviation ({TITLES_UNITS[variable]})",
                              ticks=custom_ticks)
            cb2.ax.set_yticklabels(tick_labels)
    
    # Add overall title
    plt.suptitle('Northeast Pacific Ocean Seasonal Analysis (2014-2016 vs 2011-2021)', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, 'seasonal_anomaly_analysis.png'), dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Seasonal plot saved to {os.path.join(OUTPUT_DIR, 'seasonal_anomaly_analysis.png')}")

#%% 
#--------- MAIN EXECUTION ---------#
if __name__ == "__main__":
    # Choose which plots to generate
    generate_combined_plot = True
    generate_seasonal_plot = False
    
    if generate_combined_plot:
        print("\n==== Generating combined plot for all months ====")
        plot_all_combined()
    
    if generate_seasonal_plot:
        print("\n==== Generating seasonal plots ====")
        plot_all_combined_seasonal()

