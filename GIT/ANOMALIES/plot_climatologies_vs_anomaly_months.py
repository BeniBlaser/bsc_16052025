"""
Plots anomalies of chosen variables in monthly resolution

Author: Beni Blaser
This script was created with assitance from Claude Sonnet 3.7, ran on Copilot. It helped creating, debugging and commenting the code.
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

# MODIFIED: Single month to analyze (January 2014)
TARGET_YEAR = 2014
TARGET_MONTH = "001"  # January

# Years for baseline calculation (excluding target year)
REFERENCE_YEARS = list(range(2011, 2021 + 1))
if TARGET_YEAR in REFERENCE_YEARS:
    REFERENCE_YEARS.remove(TARGET_YEAR)

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
# MODIFIED: Load January climatology for a specific variable
def load_single_month_climatology(variable, month_str):
    """
    Load climatology data for a given variable and specific month.
    """
    print(f"Loading {month_str} climatology data for {variable}...")
    
    # Load climatology file for the target month
    ds = load_climatology_file(month_str, variable)
    
    if ds is None or variable not in ds:
        print(f"  Warning: Could not load {variable} for month {month_str}")
        return None
        
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
            
    # Close dataset to free memory
    ds.close()
    del ds
    gc.collect()
    
    print(f"Successfully loaded {month_str} climatology for {variable}")
    return depth_avg

# MODIFIED: Load single month data and calculate anomaly
def load_single_month_anomaly(variable, year, month_str):
    """
    Load single month data and calculate anomaly against the monthly climatology.
    """
    print(f"Loading {month_str} {year} data for {variable} and calculating anomaly...")
    
    # Load monthly climatology for the target month
    monthly_clim = load_single_month_climatology(variable, month_str)
    if monthly_clim is None:
        return None
    
    # Load actual data file for the target month and year
    ds = load_data_file(year, month_str, variable)
    
    if ds is None or variable not in ds:
        print(f"  Warning: Could not load {variable} for {year}/{month_str}")
        return None
    
    # Get depth levels appropriate for this variable
    depth_levels = get_depth_levels(variable)
    
    # Calculate depth average for actual data
    month_depth_avg = compute_depth_average(ds, variable, depth_levels)
    
    if month_depth_avg is not None:
        # If file has daily data, calculate monthly mean
        if 'time' in month_depth_avg.dims:
            month_mean = month_depth_avg.mean(dim='time')
        else:
            month_mean = month_depth_avg
        
        # Calculate anomaly against climatology
        anomaly = month_mean - monthly_clim
        
        # Close dataset to free memory
        ds.close()
        del ds
        gc.collect()
        
        print(f"Successfully calculated anomaly for {variable} in {month_str} {year}")
        return anomaly
    
    # Close dataset to free memory
    ds.close()
    del ds
    gc.collect()
    
    print(f"Error: Could not calculate anomaly for {variable} in {month_str} {year}")
    return None

#%%
#--------- PLOTTING FUNCTIONS ---------#
def plot_single_month_anomaly():
    """
    Plot all variables in a 1×3 grid with anomalies for the selected month.
    """
    n = len(VARIABLES)
    print(f"\nCreating anomaly plot for {TARGET_MONTH} {TARGET_YEAR} vs. climatology...")
    
    # Adjust figure size for 1x3 grid (reduced height)
    fig = plt.figure(figsize=(26, 8), dpi=300)
    
    # Create a 1xn grid instead of 2xn
    gs = fig.add_gridspec(1, n, width_ratios=[1, 1, 1], wspace=0)
    
    # Get the study region bounds
    region = get_study_region()
    
    # Convert longitudes to -180 to 180 format for Cartopy
    lon_min_180 = region['lon_min'] - 360 if region['lon_min'] > 180 else region['lon_min']
    lon_max_180 = region['lon_max'] - 360 if region['lon_max'] > 180 else region['lon_max']
    
    for i, variable in enumerate(VARIABLES):
        print(f"  Processing {variable}...")
        
        # Create anomaly axis only using the GridSpec
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
        
        # Load data for the specific month
        # Still need to load both climatology and anomaly data
        clim = load_single_month_climatology(variable, TARGET_MONTH)
        anom = load_single_month_anomaly(variable, TARGET_YEAR, TARGET_MONTH)
        
        if clim is None or anom is None:
            print(f"  Error: Could not load data for {variable}")
            continue
        
        # Apply unit conversion
        conversion_factor = UNIT_CONVERSION[variable]
        if conversion_factor != 1.0:
            clim = clim * conversion_factor
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
        
        # Get colormaps for anomaly
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
        
        # Compute data array for anomaly
        anom_vals = anom.compute().values
        
        # Plot with proper transform
        if use_2d:
            # When using 2D coordinates, ensure they're in proper format for plotting
            plot_lon = lon.values - 360 if np.any(lon.values > 180) else lon.values
            pc2 = axa.pcolormesh(plot_lon, lat, anom_vals, cmap=discrete_anom_cmap, 
                              norm=anom_norm, transform=ccrs.PlateCarree())
        else:
            # For 1D coordinate arrays or direct plotting
            pc2 = axa.pcolormesh(anom_vals, cmap=discrete_anom_cmap, norm=anom_norm,
                             transform=ccrs.PlateCarree())
        
        # Convert month number to name
        month_names = {
            "001": "January", "002": "February", "003": "March",
            "004": "April", "005": "May", "006": "June",
            "007": "July", "008": "August", "009": "September",
            "010": "October", "011": "November", "012": "December"
        }
        month_name = month_names.get(TARGET_MONTH, TARGET_MONTH)
        
        # Add title for anomaly plot
        axa.set_title(f"{TITLES[variable]} Anomaly ({month_name} {TARGET_YEAR} vs Climatology)\n({depth_label})")
        
        # Create proper colorbar with consistent height
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
    
    # Save figure with tight layout to further reduce whitespace
    output_file = os.path.join(OUTPUT_DIR, f'single_month_anomaly_{TARGET_YEAR}_{TARGET_MONTH}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close(fig)

#%% 
#--------- MAIN EXECUTION ---------#
if __name__ == "__main__":
    print(f"\n==== Generating single month anomaly plot for {TARGET_MONTH}/{TARGET_YEAR} ====")
    plot_single_month_anomaly()

def plot_all_months_anomaly_2014():
    """
    Plot all variables for every month in 2014 in a 12×3 grid (12 months × 3 variables).
    Only showing anomalies (no climatology).
    """
    n_vars = len(VARIABLES)
    n_months = 12  # All months of 2014
    print(f"\nCreating anomaly plots for all months in 2014...")
    
    # Create a large figure to accommodate 12×3 grid
    fig = plt.figure(figsize=(27, 38), dpi=300)
    
    # Organize as 12 rows (months) × 3 columns (variables)
    gs = fig.add_gridspec(n_months, n_vars, wspace=0.1, hspace=0.3)
    
    # Get the study region bounds
    region = get_study_region()
    
    # Convert longitudes to -180 to 180 format for Cartopy
    lon_min_180 = region['lon_min'] - 360 if region['lon_min'] > 180 else region['lon_min']
    lon_max_180 = region['lon_max'] - 360 if region['lon_max'] > 180 else region['lon_max']
    
    # Month names mapping
    month_names = {
        "001": "January", "002": "February", "003": "March",
        "004": "April", "005": "May", "006": "June",
        "007": "July", "008": "August", "009": "September",
        "010": "October", "011": "November", "012": "December"
    }
    
    # Process each month
    for m_idx, month_num in enumerate(range(1, 13)):  # All 12 months
        month_str = f"{month_num:03d}"
        month_name = month_names.get(month_str, month_str)
        print(f"Processing {month_name} 2014...")
        
        # Process each variable for this month
        for v_idx, variable in enumerate(VARIABLES):
            print(f"  Processing {variable}...")
            
            # Create axis with proper projection
            ax = fig.add_subplot(gs[m_idx, v_idx], projection=ccrs.PlateCarree(central_longitude=0))
            
            # Get depth level information and create proper depth label
            depth_levels = get_depth_levels(variable)
            if variable == 'POC_FLUX_IN':
                depth_label = f"at {abs(depth_levels[0])}m"
            else:
                depth_label = f"0m-{abs(min(depth_levels))}m mean"
            
            # Set the map extent using the converted coordinates
            ax.set_extent([lon_min_180, lon_max_180, 
                          region['lat_min'], region['lat_max']], 
                         crs=ccrs.PlateCarree())
            
            # Add coastlines and land features
            ax.coastlines(resolution='50m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            
            # Add gridlines with proper configuration
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                             alpha=0.5, linestyle='--')
            
            gl.xlocator = mticker.FixedLocator(range(-160, -100, 10))
            gl.ylocator = mticker.FixedLocator(range(20, 70, 10))
            
            # Only show bottom and left labels except for specific positions
            gl.top_labels = False
            gl.right_labels = False
            
            # Only show bottom labels on the bottom row
            if m_idx < n_months - 1:
                gl.bottom_labels = False
                
            # Only show left labels on the first column 
            if v_idx > 0:
                gl.left_labels = False
                
            # Ensure label styles are set to make ticks visible
            gl.xlabel_style = {'size': 10, 'color': 'black'}
            gl.ylabel_style = {'size': 10, 'color': 'black'}
            
            # Load data for this specific month
            clim = load_single_month_climatology(variable, month_str)
            anom = load_single_month_anomaly(variable, TARGET_YEAR, month_str)
            
            if clim is None or anom is None:
                print(f"  Error: Could not load data for {variable}, {month_name}")
                # Add text to indicate missing data
                ax.text(0.5, 0.5, f"No data available", 
                      transform=ax.transAxes, ha='center', fontsize=12)
                continue
            
            # Apply unit conversion
            conversion_factor = UNIT_CONVERSION[variable]
            if conversion_factor != 1.0:
                clim = clim * conversion_factor
                anom = anom * conversion_factor
            
            # Calculate anomaly limits from the 99th percentile
            abs_anom = np.abs(anom.values)
            abs_anom = abs_anom[~np.isnan(abs_anom)]
            
            # Calculate 99th percentile value
            if len(abs_anom) > 0:
                vmax_a = np.percentile(abs_anom, 99)
            else:
                vmax_a = VLIMS[variable]['anom'][1]
                
            # Set symmetric limits
            vmin_a = -vmax_a
            
            # Get coordinates
            lon = clim.coords.get('lon', None)
            lat = clim.coords.get('lat', None)
            use_2d = lon is not None and lon.ndim == 2
            
            # Get colormap for anomaly
            if variable == 'temp':
                anom_cmap_base = cmo.cm.balance
            elif variable == 'TOT_PROD':
                anom_cmap_base = cmo.cm.tarn
            else:
                anom_cmap_base = cmo.cm.diff_r
                
            # Create discrete colormap for anomalies
            discrete_anom_cmap = create_symmetric_discrete_colormap(anom_cmap_base, levels=9)
            
            # Create boundary norm based on calculated limits
            levels = 9
            anom_bounds = np.linspace(vmin_a, vmax_a, levels + 1)
            anom_norm = BoundaryNorm(anom_bounds, discrete_anom_cmap.N)
            
            # Compute data array for anomaly
            anom_vals = anom.compute().values
            
            # Plot with proper transform
            if use_2d:
                # When using 2D coordinates, ensure proper format for plotting
                plot_lon = lon.values - 360 if np.any(lon.values > 180) else lon.values
                pc = ax.pcolormesh(plot_lon, lat, anom_vals, cmap=discrete_anom_cmap, 
                                 norm=anom_norm, transform=ccrs.PlateCarree())
            else:
                # For 1D coordinate arrays or direct plotting
                pc = ax.pcolormesh(anom_vals, cmap=discrete_anom_cmap, norm=anom_norm,
                                 transform=ccrs.PlateCarree())
            
            # Add title for this specific plot - simplified for grid layout
            if m_idx == 0:  # First row gets variable names
                ax.set_title(f"{TITLES[variable]}\n{depth_label}", fontsize=12)
            
            # Add month label on the left side
            if v_idx == 0:  # First column gets month names
                ax.text(-0.1, 0.5, month_name, va='center', ha='right',
                       transform=ax.transAxes, fontsize=14, fontweight='bold')
            
            # Add colorbar for each plot
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
            
            # Use bounds as tick positions for anomaly colorbar
            cbar = fig.colorbar(pc, cax=cax, ticks=anom_bounds)
            
            # Only first row gets colorbar labels
            if m_idx == 0:  
                cbar.set_label(f"Δ {TITLES_UNITS[variable]}", size=10)
            
            # Format anomaly tick labels with appropriate precision
            if variable == 'temp':
                cbar.ax.set_yticklabels([f"{t:.1f}" for t in anom_bounds], fontsize=8)
            elif variable == 'TOT_PROD':
                cbar.ax.set_yticklabels([f"{t:.2f}" for t in anom_bounds], fontsize=8)
            else:
                cbar.ax.set_yticklabels([f"{t:.1f}" for t in anom_bounds], fontsize=8)
            
            # Ensure ticks are visible
            cbar.ax.tick_params(width=1, length=3)
    
    # Save figure
    output_file = os.path.join(OUTPUT_DIR, f'monthly_anomalies_2014_all_months.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close(fig)

#--------- MAIN EXECUTION ---------#
if __name__ == "__main__":
    print(f"\n==== Generating monthly anomaly plots for all months of 2014 ====")
    plot_all_months_anomaly_2014()

def plot_all_months_multi_year_anomaly():
    """
    Plot all variables for every month in 2014-2016 in a 12×9 grid.
    Uses consistent color scales across all years for each variable.
    """
    n_vars = len(VARIABLES)
    n_months = 12  # All months
    years = [2014, 2015, 2016]  # Three years to plot
    n_years = len(years)
    
    print(f"\nCreating anomaly plots for all months in 2014-2016...")
    
    # Step 1: Pre-calculate global min/max values for each variable across all months/years
    print("Calculating global percentile values for consistent color scales...")
    global_values = {var: [] for var in VARIABLES}
    
    # First collect all anomaly values
    for month_num in range(1, 13):
        month_str = f"{month_num:03d}"
        for year in years:
            for variable in VARIABLES:
                # Load data for this specific month and year
                clim = load_single_month_climatology(variable, month_str)
                anom = load_single_month_anomaly(variable, year, month_str)
                
                if clim is None or anom is None:
                    continue
                
                # Apply unit conversion
                conversion_factor = UNIT_CONVERSION[variable]
                if conversion_factor != 1.0:
                    anom = anom * conversion_factor
                
                # Get actual values, removing NaNs
                anom_vals = anom.values
                valid_vals = anom_vals[~np.isnan(anom_vals)]
                
                # Add to our collection
                if len(valid_vals) > 0:
                    global_values[variable].extend(valid_vals.flatten())
    
    # Calculate appropriate percentiles for each variable
    global_limits = {}
    for variable in VARIABLES:
        if len(global_values[variable]) > 0:
            if variable == 'temp':
                # For temperature, use the 99th percentile
                p99 = np.percentile(global_values[variable], 99)
                global_limits[variable] = [-p99, p99]
                print(f"  Temperature: Using 99th percentile = {p99:.4f}°C (symmetric)")
            else:
                # For POC_FLUX_IN and TOT_PROD, use the 1st percentile
                p01 = np.abs(np.percentile(global_values[variable], 1))
                global_limits[variable] = [-p01, p01]
                print(f"  {variable}: Using 1st percentile = {p01:.4f} (symmetric)")
        else:
            # Fallback
            global_limits[variable] = [-VLIMS[variable]['anom'][1], VLIMS[variable]['anom'][1]]
            print(f"  Warning: Using default limits for {variable}")
    
    # Create a very large figure to accommodate 12×9 grid
    fig = plt.figure(figsize=(60, 70), dpi=300)
    
    # Organize as 12 rows (months) × 9 columns (3 variables × 3 years)
    gs = fig.add_gridspec(n_months, n_vars * n_years, wspace=0.05, hspace=0.25)
    
    # Get the study region bounds
    region = get_study_region()
    
    # Convert longitudes to -180 to 180 format for Cartopy
    lon_min_180 = region['lon_min'] - 360 if region['lon_min'] > 180 else region['lon_min']
    lon_max_180 = region['lon_max'] - 360 if region['lon_max'] > 180 else region['lon_max']
    
    # Month names mapping
    month_names = {
        "001": "January", "002": "February", "003": "March",
        "004": "April", "005": "May", "006": "June",
        "007": "July", "008": "August", "009": "September",
        "010": "October", "011": "November", "012": "December"
    }
    
    # Calculate column indices for year labels
    year_col_indices = [n_vars * y + n_vars // 2 for y in range(n_years)]
    
    # Add year labels at the top
    for y_idx, year in enumerate(years):
        col_start = n_vars * y_idx
        col_span = n_vars
        ax_year = fig.add_subplot(gs[0, col_start:col_start+col_span])
        ax_year.text(0.5, 0.5, f"{year}", ha='center', va='center', fontsize=24, fontweight='bold')
        ax_year.axis('off')  # Hide the axis
    
    # Process each month
    for m_idx, month_num in enumerate(range(1, 13)):  # All 12 months
        month_str = f"{month_num:03d}"
        month_name = month_names.get(month_str, month_str)
        print(f"Processing {month_name} for years 2014-2016...")
        
        # Process each year
        for y_idx, year in enumerate(years):
            current_year = year
            
            # Process each variable for this month and year
            for v_idx, variable in enumerate(VARIABLES):
                # Calculate the column index based on year and variable
                col_idx = y_idx * n_vars + v_idx
                
                print(f"  Processing {month_name} {current_year} - {variable}...")
                
                # Create axis with proper projection
                ax = fig.add_subplot(gs[m_idx, col_idx], projection=ccrs.PlateCarree(central_longitude=0))
                
                # Get depth level information and create proper depth label
                depth_levels = get_depth_levels(variable)
                if variable == 'POC_FLUX_IN':
                    depth_label = f"at {abs(depth_levels[0])}m"
                else:
                    depth_label = f"0m-{abs(min(depth_levels))}m mean"
                
                # Set the map extent using the converted coordinates
                ax.set_extent([lon_min_180, lon_max_180, 
                              region['lat_min'], region['lat_max']], 
                             crs=ccrs.PlateCarree())
                
                # Add coastlines and land features
                ax.coastlines(resolution='50m', color='black', linewidth=0.5)
                ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
                
                # Add gridlines with proper configuration
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                                 alpha=0.5, linestyle='--')
                
                gl.xlocator = mticker.FixedLocator(range(-160, -100, 20))  # Wider spacing
                gl.ylocator = mticker.FixedLocator(range(20, 70, 20))  # Wider spacing
                
                # Only show bottom and left labels except for specific positions
                gl.top_labels = False
                gl.right_labels = False
                
                # Only show bottom labels on the bottom row
                if m_idx < n_months - 1:
                    gl.bottom_labels = False
                    
                # Only show left labels on the first column of each year group
                if v_idx > 0:
                    gl.left_labels = False
                    
                # Ensure label styles are set to make ticks visible
                gl.xlabel_style = {'size': 8, 'color': 'black'}
                gl.ylabel_style = {'size': 8, 'color': 'black'}
                
                # Load data for this specific month and year
                clim = load_single_month_climatology(variable, month_str)
                anom = load_single_month_anomaly(variable, current_year, month_str)
                
                if clim is None or anom is None:
                    print(f"  Error: Could not load data for {variable}, {month_name} {current_year}")
                    # Add text to indicate missing data
                    ax.text(0.5, 0.5, f"No data available", 
                          transform=ax.transAxes, ha='center', fontsize=10)
                    continue
                
                # Apply unit conversion
                conversion_factor = UNIT_CONVERSION[variable]
                if conversion_factor != 1.0:
                    clim = clim * conversion_factor
                    anom = anom * conversion_factor
                
                # Use the pre-calculated global limits for consistent coloring
                vmin_a = global_limits[variable][0]
                vmax_a = global_limits[variable][1]
                
                # Get coordinates
                lon = clim.coords.get('lon', None)
                lat = clim.coords.get('lat', None)
                use_2d = lon is not None and lon.ndim == 2
                
                # Get colormap for anomaly
                if variable == 'temp':
                    anom_cmap_base = cmo.cm.balance
                elif variable == 'TOT_PROD':
                    anom_cmap_base = cmo.cm.tarn
                else:
                    anom_cmap_base = cmo.cm.diff_r
                    
                # Create discrete colormap for anomalies
                discrete_anom_cmap = create_symmetric_discrete_colormap(anom_cmap_base, levels=9)
                
                # Create boundary norm based on calculated global limits
                levels = 9
                anom_bounds = np.linspace(vmin_a, vmax_a, levels + 1)
                anom_norm = BoundaryNorm(anom_bounds, discrete_anom_cmap.N)
                
                # Compute data array for anomaly
                anom_vals = anom.compute().values
                
                # Plot with proper transform
                if use_2d:
                    # When using 2D coordinates, ensure proper format for plotting
                    plot_lon = lon.values - 360 if np.any(lon.values > 180) else lon.values
                    pc = ax.pcolormesh(plot_lon, lat, anom_vals, cmap=discrete_anom_cmap, 
                                     norm=anom_norm, transform=ccrs.PlateCarree())
                else:
                    # For 1D coordinate arrays or direct plotting
                    pc = ax.pcolormesh(anom_vals, cmap=discrete_anom_cmap, norm=anom_norm,
                                     transform=ccrs.PlateCarree())
                
                # Add variable titles only in the first row (after year labels)
                if m_idx == 0:
                    # Add variable title in first row
                    if v_idx == 0:  # First variable in group
                        title = f"{TITLES[variable]}\n({depth_label})"
                    else:  # Other variables in group
                        title = f"{TITLES[variable]}\n({depth_label})"
                    ax.set_title(title, fontsize=10)
                
                # Add month label on the left side of first column only
                if col_idx == 0:  # First column gets month names
                    ax.text(-0.2, 0.5, month_name, va='center', ha='right',
                           transform=ax.transAxes, fontsize=12, fontweight='bold')
                
                # Add colorbar for each plot
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
                
                # Use bounds as tick positions for anomaly colorbar
                cbar = fig.colorbar(pc, cax=cax, ticks=anom_bounds)
                
                # Only plots in the first row (after year labels) get colorbar labels
                if m_idx == 0:  
                    cbar.set_label(f"Δ {TITLES_UNITS[variable]}", size=8)
                
                # Format anomaly tick labels with appropriate precision
                if variable == 'temp':
                    cbar.ax.set_yticklabels([f"{t:.1f}" for t in anom_bounds], fontsize=6)
                elif variable == 'TOT_PROD':
                    cbar.ax.set_yticklabels([f"{t:.2f}" for t in anom_bounds], fontsize=6)
                else:
                    cbar.ax.set_yticklabels([f"{t:.1f}" for t in anom_bounds], fontsize=6)
                
                # Ensure ticks are visible
                cbar.ax.tick_params(width=1, length=3)
    
    # Add overall title
    plt.suptitle(f'Northeast Pacific Ocean - Monthly Anomalies (2014-2016) vs Climatology (2011-2021)', 
                fontsize=32, y=0.85)
    
    # Save figure
    output_file = os.path.join(OUTPUT_DIR, f'monthly_anomalies_2014_2016_all_months.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close(fig)

#--------- MAIN EXECUTION ---------#
if __name__ == "__main__":
    print(f"\n==== Generating monthly anomaly plots for 2014-2016 ====")
    plot_all_months_multi_year_anomaly()
