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
from matplotlib.patches import Rectangle

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
    """
    from matplotlib.colors import ListedColormap
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
    
    return cmap_discrete

# Function to convert longitude from -180 to 180 format to 0 to 360 format
def convert_lon_to_360(lon):
    """Convert longitude from -180 to 180 format to 0 to 360 format."""
    return lon + 360 if lon < 0 else lon

# Function to convert longitude from 0 to 360 format to -180 to 180 format
def convert_lon_to_180(lon):
    """Convert longitude from 0 to 360 format to -180 to 180 format."""
    return lon - 360 if lon > 180 else lon

# Define study regions - these are in -180 to 180 format
STUDY_REGIONS = [
    {
        'name': 'gulf_alaska',
        'lon_min': -153, 'lon_max': -135,
        'lat_min': 55, 'lat_max': 60
    },    

    {
        'name': 'lownpp_north',
        'lon_min': -131., 'lon_max': -127,
        'lat_min': 42, 'lat_max': 47.25
    },
    {
        'name': 'highnpp_north',
        'lon_min': -126.5, 'lon_max': -124,
        'lat_min': 43, 'lat_max': 49
    },
    {
        'name': 'low_npp_northerngyre',
        'lon_min': -133.5, 'lon_max': -131.5,
        'lat_min': 49.5, 'lat_max': 52.5
    },
    {
        'name': 'lownpp_central',
        'lon_min': -130, 'lon_max': -124,
        'lat_min': 35.5, 'lat_max': 38.5
    },
    {
        'name': 'highnpp_central',
        'lon_min': -131, 'lon_max': -125,
        'lat_min': 39, 'lat_max': 41.5
    },

    {
        'name': 'lownpp_south',
        'lon_min': -122.5, 'lon_max': -118,
        'lat_min': 33, 'lat_max': 35.25
    },
    {
        'name': 'highnpp_south',
        'lon_min': -123, 'lon_max': -120.75,
        'lat_min': 33, 'lat_max': 31
    },
]

# Convert and store study regions in 0-360 format
REGIONS_360 = {}
for i, region in enumerate(STUDY_REGIONS):
    region_id = str(i+1)
    REGIONS_360[region_id] = {
        'name': region['name'],
        'lon_min': convert_lon_to_360(region['lon_min']),
        'lon_max': convert_lon_to_360(region['lon_max']),
        'lat_min': region['lat_min'],
        'lat_max': region['lat_max']
    }

# Define the overall study region - this is already in 0-360 format
STUDY_REGION = {
    "lon_min": 205, "lon_max": 245,  # Full longitude range (0-360 format)
    "lat_min": 20, "lat_max": 62,    # Full latitude range
    "name": "Northeast Pacific"
}

# Color mapping for different regions
REGION_COLORS = {

    "gulf_alaska": "blue",

    "highnpp_south": "green",        
    "highnpp_central": "green",    
    "highnpp_north": "green",
    

    "lownpp_south": "red", 
    "low_npp_northerngyre": "red",
    "lownpp_central": "red",
    "lownpp_north": "red",
}

def add_regions_to_map(ax, use_360=False, show_labels=True, linewidth=1.5):
    """
    Add region rectangles to an existing map axes.
    """
    # Determine which regions format to use based on the map's coordinate system
    regions = REGIONS_360 if use_360 else STUDY_REGIONS
    transform = ccrs.PlateCarree()
    
    # Handle different structures based on whether regions is a dict or list
    if isinstance(regions, dict):
        # For dictionary structure (REGIONS_360)
        for region_id, region_info in regions.items():
            if isinstance(region_id, str) and region_id.isdigit():
                # Get region name from the info dictionary
                region_name = region_info['name']
            else:
                # Use region_id as the name if it's not a number
                region_name = region_id
                
            # Get color for this region
            color = REGION_COLORS.get(region_name, "royalblue")
            
            # Get region bounds
            if use_360:
                # Map is in 0-360 format, use bounds directly
                lon_min = region_info['lon_min']
                lon_max = region_info['lon_max']
            else:
                # Map is in -180-180 format, need to convert if regions are in 0-360
                if 'lon_min' in region_info:
                    lon_min = convert_lon_to_180(region_info['lon_min']) if region_info['lon_min'] > 180 else region_info['lon_min']
                    lon_max = convert_lon_to_180(region_info['lon_max']) if region_info['lon_max'] > 180 else region_info['lon_max']
                else:
                    # Direct access for STUDY_REGIONS format
                    lon_min = region_info['lon_min']
                    lon_max = region_info['lon_max']
                    
            lat_min = region_info['lat_min']
            lat_max = region_info['lat_max']
            
            # Add rectangle and label
            width = lon_max - lon_min
            height = lat_max - lat_min
            rect = Rectangle(
                (lon_min, lat_min), width, height,
                edgecolor=color, facecolor='none',
                linewidth=linewidth, alpha=0.8,
                transform=transform
            )
            ax.add_patch(rect)
            
            if show_labels:
                center_lon = (lon_min + lon_max) / 2
                center_lat = (lat_min + lat_max) / 2
                
                ax.text(
                    center_lon, center_lat, region_name,
                    color='black', fontweight='bold', ha='center', va='center',
                    transform=transform,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', pad=0.3)
                )
    
    else:
        # For list structure (STUDY_REGIONS)
        for i, region_info in enumerate(regions):
            region_name = region_info['name']
            color = REGION_COLORS.get(region_name, "royalblue")
            
            # Get region bounds
            if use_360:
                # Map is in 0-360 format, use bounds directly
                lon_min = region_info['lon_min']
                lon_max = region_info['lon_max']
            else:
                # Map is in -180-180 format
                lon_min = region_info['lon_min']
                lon_max = region_info['lon_max']
                
            lat_min = region_info['lat_min']
            lat_max = region_info['lat_max']
            
            # Add rectangle and label
            width = lon_max - lon_min
            height = lat_max - lat_min
            rect = Rectangle(
                (lon_min, lat_min), width, height,
                edgecolor=color, facecolor='none',
                linewidth=linewidth, alpha=0.8,
                transform=transform
            )
            ax.add_patch(rect)
            
            if show_labels:
                center_lon = (lon_min + lon_max) / 2
                center_lat = (lat_min + lat_max) / 2
                
                ax.text(
                    center_lon, center_lat, region_name,
                    color='black', fontweight='bold', ha='center', va='center',
                    transform=transform,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', pad=0.3)
                )

def create_region_map(output_dir):
    """Create a standalone map of the Northeast Pacific with all regions marked as rectangles."""
    print("\nCreating region map...")
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set map extent - convert to -180 to 180 for Cartopy
    lon_min_180 = convert_lon_to_180(STUDY_REGION["lon_min"])
    lon_max_180 = convert_lon_to_180(STUDY_REGION["lon_max"])
    
    ax.set_extent([lon_min_180, lon_max_180, 
                  STUDY_REGION["lat_min"], STUDY_REGION["lat_max"]], 
                  crs=ccrs.PlateCarree())
    
    # Add natural earth features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)
    
    # Add grid lines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                     alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add all regions to the map (using -180 to 180 format)
    add_regions_to_map(ax, use_360=False, show_labels=True, linewidth=2)
    
    # Add title
    ax.set_title(f"Analysis Regions - {STUDY_REGION['name']}", fontsize=14)
    
    # Save figure
    map_output = os.path.join(output_dir, "region_map.png")
    plt.savefig(map_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Region map saved to: {map_output}")

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

# Depth configurations
DEPTH_LEVELS_50M = [0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50]
POC_FLUX_DEPTH = -100  # Use 100m depth for POC_FLUX_IN

# Plot configurations
COLORMAPS = {
    'temp': cmo.cm.amp,
    'TOT_PROD': cmo.cm.tempo,
    'POC_FLUX_IN': create_blue_diff_cmap()
}

# Value ranges for colormaps (fallback values)
VLIMS = {
    'temp': {'clim': [5, 25], 'anom': [-1.2, 1.2]},
    'TOT_PROD': {'clim': [0, 0.6], 'anom': [-0.0000025 * 31536.0, 0.0000025 * 31536.0]},
    'POC_FLUX_IN': {'clim': [0, 6.0], 'anom': [-0.00003 * 31536.0, 0.00003 * 31536.0]}
}

# Plot titles and labels
TITLES = {
    'temp': 'Temperature',
    'TOT_PROD': 'Total Production',
    'POC_FLUX_IN': 'POC Flux'
}

# Define conversion factors for the units
UNIT_CONVERSION = {
    'temp': 1.0,  # No conversion needed
    'TOT_PROD': 31536.0,  # Convert mmol/m^3/s to mol/m^3/year
    'POC_FLUX_IN': 31536.0  # Convert mmolC/m^2/s to molC/m^2/year
}

# Units for titles
TITLES_UNITS = {
    'temp': '°C',
    'TOT_PROD': 'mol m⁻³ yr⁻¹',
    'POC_FLUX_IN': 'mol C m⁻² yr⁻¹'
}

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
    
    # Loop through all months (5-10, May-October)
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

#%%
#--------- PLOTTING FUNCTION ---------#
def plot_anomalies_only():
    """Plot only the anomalies for all variables in a 1×3 grid with region labels and callouts."""
    n = len(VARIABLES)
    print("\nCreating anomalies-only plot with data-driven limits, discrete colormap, and region labels...")
    
    # Adjust figure size for a single row
    fig = plt.figure(figsize=(26, 8), dpi=300)
    
    # Create gridspec for a single row
    gs = fig.add_gridspec(1, n, width_ratios=[1, 1, 1], wspace=0.4)
    
    # Get the study region bounds
    region = get_study_region()
    
    # Convert longitudes to -180 to 180 format for Cartopy
    lon_min_180 = region['lon_min'] - 360 if region['lon_min'] > 180 else region['lon_min']
    lon_max_180 = region['lon_max'] - 360 if region['lon_max'] > 180 else region['lon_max']
    
    # Only add labels to the first subplot
    add_region_labels = True
    
    for i, variable in enumerate(VARIABLES):
        print(f"  Processing {variable}...")
        
        # Create axes with proper projection using the GridSpec - only anomaly axes
        axa = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree(central_longitude=0))
        
        # Get depth level information and create proper depth label
        depth_levels = get_depth_levels(variable)
        if variable == 'POC_FLUX_IN':
            # For POC_FLUX_IN, use "at 100m" label
            depth_label = f"at {abs(depth_levels[0])}m"
        else:
            # For other variables with depth range, use "0m-50m" format
            depth_label = f"0m-{abs(min(depth_levels))}m mean"
        
        # Set the map extent using the converted coordinates
        axa.set_extent([lon_min_180, lon_max_180, 
                        region['lat_min'], region['lat_max']], 
                       crs=ccrs.PlateCarree())
        
        # Add coastlines and land features
        axa.coastlines(resolution='50m', color='black', linewidth=0.5)
        axa.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        
        # Add gridlines with proper configuration for tick display
        gl2 = axa.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                          alpha=0.5, linestyle='--')
        
        # Configure grid lines with explicit tick locations
        gl2.xlocator = mticker.FixedLocator(range(-160, -100, 10))
        gl2.ylocator = mticker.FixedLocator(range(20, 70, 10))
        
        # Turn off right labels (and top labels for cleaner look)
        gl2.right_labels = False 
        gl2.top_labels = False
        
        # Ensure label styles are set to make ticks visible
        gl2.xlabel_style = {'size': 12, 'color': 'black'}
        gl2.ylabel_style = {'size': 12, 'color': 'black'}
        
        # Load data
        clim = load_climatology_data(variable)  # Still need this for coordinates
        anom = load_anomaly_data(variable, tuple(ANOMALY_YEARS))
        if clim is None or anom is None:
            print(f"  Error: Could not load data for {variable}")
            continue
        
        # Apply unit conversion
        conversion_factor = UNIT_CONVERSION[variable]
        if conversion_factor != 1.0:
            anom = anom * conversion_factor
        
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
        
        # Get colormaps for anomalies
        if variable == 'temp':
            anom_cmap_base = cmo.cm.balance
        elif variable == 'TOT_PROD':
            anom_cmap_base = cmo.cm.tarn
        else:
            anom_cmap_base = cmo.cm.diff_r
            
        # Create discrete colormap for anomalies
        discrete_anom_cmap = create_symmetric_discrete_colormap(anom_cmap_base, levels=9)
        
        # Create boundary norm based on the calculated limits
        levels = 9
        anom_bounds = np.linspace(vmin_a, vmax_a, levels + 1)
        anom_norm = BoundaryNorm(anom_bounds, discrete_anom_cmap.N)
        
        # Compute data arrays
        anom_vals = anom.compute().values
        
        # Plot with proper transform
        if use_2d:
            # When using 2D coordinates, ensure they're in proper format for plotting
            plot_lon = lon.values - 360 if np.any(lon.values > 180) else lon.values
            pc2 = axa.pcolormesh(plot_lon, lat, anom_vals, cmap=discrete_anom_cmap, 
                              norm=anom_norm, transform=ccrs.PlateCarree(), alpha=0.7)  # Added alpha
        else:
            # For 1D coordinate arrays or direct plotting
            pc2 = axa.pcolormesh(anom_vals, cmap=discrete_anom_cmap, norm=anom_norm,
                             transform=ccrs.PlateCarree(), alpha=0.7)  # Added alpha
        
        # Add title
        axa.set_title(f"{TITLES[variable]} Anomaly (2014-2016 vs Baseline Period)\n({depth_label})")
        
        # Create proper colorbars with consistent height
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # For anomaly colorbar
        divider2 = make_axes_locatable(axa)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
        
        # Use bounds as tick positions for anomaly colorbar
        cbar2 = fig.colorbar(pc2, cax=cax2, ticks=anom_bounds)
        cbar2.set_label(f"Δ {TITLES_UNITS[variable]}", size=12)
        
        # Format anomaly tick labels with appropriate precision
        if variable == 'temp':
            cbar2.ax.set_yticklabels([f"{t:.1f}" for t in anom_bounds])
        elif variable == 'TOT_PROD':
            cbar2.ax.set_yticklabels([f"{t:.2f}" for t in anom_bounds])
        else:
            cbar2.ax.set_yticklabels([f"{t:.1f}" for t in anom_bounds])
        
        # Ensure ticks are visible by setting tick parameters
        cbar2.ax.tick_params(labelsize=12, width=1, length=5)
        
        # Add region overlays without labels (we'll add them separately)
        add_regions_to_map(axa, use_360=False, show_labels=False)
        
        # Add region labels with callouts to all plots (removed i == 0 condition)
        if add_region_labels:
            # Create a dictionary to store region centers for drawing callout lines
            region_centers = {}
            for region_info in STUDY_REGIONS:
                region_name = region_info['name']
                # Get region bounds
                lon_min = region_info['lon_min']
                lon_max = region_info['lon_max']
                lat_min = region_info['lat_min']
                lat_max = region_info['lat_max']
                # Calculate center coordinates
                center_lon = (lon_min + lon_max) / 2
                center_lat = (lat_min + lat_max) / 2
                region_centers[region_name] = (center_lon, center_lat)
            
            # Create label box in bottom left corner with transparent background
            label_x = lon_min_180 + 5  # Offset from left edge
            label_y = region['lat_min'] + 3  # Offset from bottom edge
            
            # Create a text box with region names
            region_names = [region_info['name'] for region_info in STUDY_REGIONS]
            colors = [REGION_COLORS.get(name, "royalblue") for name in region_names]
            
            # Calculate vertical spacing for labels
            label_spacing = 1.8  # degrees of latitude between labels
            
            # Add each region name and callout line
            for idx, (name, color) in enumerate(zip(region_names, colors)):
                # Label position (stacked vertically)
                text_y = label_y + idx * label_spacing
                
                # Add the region name text
                text = axa.text(
                    label_x, text_y, name,
                    color=color, fontweight='bold', ha='left', va='center',
                    transform=ccrs.PlateCarree(),
                    bbox=dict(facecolor='white', alpha=1, boxstyle='square', pad=0.2, edgecolor=color)
                )
                
                # Get region center for callout line
                center_lon, center_lat = region_centers[name]
                
                # Draw callout line from label to region center
                # Use curved connector for better visual separation
                from matplotlib.patches import ConnectionPatch
                
                # Create connection patch with bezier curve
                con = ConnectionPatch(
                    xyA=(label_x + 5, text_y),  # Start slightly to the right of the text
                    xyB=(center_lon, center_lat),
                    coordsA="data", coordsB="data",
                    axesA=axa, axesB=axa,
                    arrowstyle="-|>", linestyle='-',
                    linewidth=1.2, color=color, alpha=0.8,
                    connectionstyle="arc3,rad=0.2"  # Curved line
                )
                axa.add_artist(con)
    
    # Add overall title with slightly reduced y position
    plt.suptitle('Northeast Pacific Ocean - Selected subregions for Probing', 
                fontsize=16, y=0.97)
    
    # Save figure with tight layout to further reduce whitespace
    output_file = os.path.join(OUTPUT_DIR, 'anomalies_only_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close(fig)

#%% 
#--------- MAIN EXECUTION ---------#
if __name__ == "__main__":
    # Create a standalone region map
    create_region_map(OUTPUT_DIR)
    
    # Generate the anomalies-only plot
    print("\n==== Generating anomalies-only plot ====")
    plot_anomalies_only()