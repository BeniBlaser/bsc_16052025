"""
Plot seasonal anomalies of selected variables.

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
import gc
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib import gridspec

#--------- COLORMAP HELPER FUNCTIONS ---------#
def create_symmetric_discrete_colormap(cmap_name, levels=9):
    """Creates a discrete symmetric diverging colormap with the middle level at zero."""
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
    return ListedColormap(colors)

#--------- CONFIGURATION ---------#
DATA_DIR = '/nfs/sea/work/bblaser/z_avg_meanpool_domain/'
CLIM_DIR = '/nfs/sea/work/bblaser/monthly_clim/'
OUTPUT_DIR = '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events'
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Variables to process
VARIABLES = ['temp', 'TOT_PROD', 'POC_FLUX_IN']

# Years for analysis
REFERENCE_YEARS = list(range(2011, 2021 + 1))
TARGET_YEAR = 2014  # Only 2014

# Depth configurations
DEPTH_LEVELS_50M = [0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50]
POC_FLUX_DEPTH = -100  # Use 100m depth for POC_FLUX_IN

# Variable metadata
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

# Fallback value ranges for colormaps
VLIMS = {
    'temp': {'clim': [5, 25], 'anom': [-1.2, 1.2]},
    'TOT_PROD': {'clim': [0, 0.6], 'anom': [-0.0000025 * 31536.0, 0.0000025 * 31536.0]},
    'POC_FLUX_IN': {'clim': [0, 6.0], 'anom': [-0.00003 * 31536.0, 0.00003 * 31536.0]}
}

# Units for variables
TITLES_UNITS = {
    'temp': '°C',
    'TOT_PROD': 'mol m⁻³ yr⁻¹',
    'POC_FLUX_IN': 'mol C m⁻² yr⁻¹'
}

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
    """Efficient region slicing for both Dataset and DataArray objects."""
    if dataset is None:
        return None
        
    region = get_study_region()
    
    # Identify coordinate variables
    lon_var = None
    lat_var = None
    
    for lon_name in ['lon_rho', 'lon']:
        if lon_name in dataset.data_vars or lon_name in dataset.coords:
            lon_var = lon_name
            break
            
    for lat_name in ['lat_rho', 'lat']:
        if lat_name in dataset.data_vars or lat_name in dataset.coords:
            lat_var = lat_name
            break
    
    # If we found both coordinates, promote them to coords and slice
    if lon_var and lat_var:
        # make lon/lat real coords
        dataset = dataset.assign_coords(lon=dataset[lon_var], lat=dataset[lat_var])
        lon_var, lat_var = 'lon', 'lat'
        
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
                    min_xi, max_xi = valid_indices[1].min(), max(valid_indices[1]).max()
                    
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
            return dataset
    
    return dataset

def load_climatology_file(month_str, variable=None):
    """Load a monthly climatology file directly."""
    clim_file = os.path.join(CLIM_DIR, f"climatology_month_{month_str}_2011-2021.nc")
    
    if not os.path.exists(clim_file):
        print(f"Warning: Climatology file not found: {clim_file}")
        return None
        
    try:
        ds = xr.open_dataset(clim_file)
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

def load_single_month_climatology(variable, month_str):
    """Load climatology data for a given variable and specific month."""
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
    
    return depth_avg

def load_single_month_anomaly(variable, year, month_str):
    """Load single month data and calculate anomaly against the monthly climatology."""
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
        
        return anomaly
    
    # Close dataset to free memory
    ds.close()
    del ds
    gc.collect()
    
    print(f"Error: Could not calculate anomaly for {variable} in {month_str} {year}")
    return None

def plot_seasonal_2014_anomalies():
    """
    Plot May-July and Aug-Oct 2014 seasonal mean anomalies in a single 2×3 grid.
    The grid contains 2 rows (seasons) × 3 columns (variables).
    """
    n_vars = len(VARIABLES)
    year = TARGET_YEAR  # 2014
    
    # Define variable display names for titles
    var_display_names = {
        'temp': 'Temperature',
        'TOT_PROD': 'NPP',
        'POC_FLUX_IN': 'POC flux'
    }
    
    # Define seasons to process
    seasons = [
        {"name": "May-Jul", "months": [5, 6, 7]},
        {"name": "Aug-Oct", "months": [8, 9, 10]}
    ]
    
    # Create a single figure for all seasons (2 rows × 3 columns)
    fig = plt.figure(figsize=(22, 13), dpi=300)
    
    # Add year as the overall title for the figure
    fig.suptitle(f"{year}", fontsize=20, y=0.98)
    
    # Create main grid - 2 rows (seasons) × 1 column
    main_gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    # Get the study region bounds
    region = get_study_region()
    lon_min_180 = region['lon_min'] - 360 if region['lon_min'] > 180 else region['lon_min']
    lon_max_180 = region['lon_max'] - 360 if region['lon_max'] > 180 else region['lon_max']
    
    # Process each season (row)
    for s_idx, season in enumerate(seasons):
        season_name = season["name"]
        month_list = season["months"]
        
        print(f"\nProcessing {season_name} {year}...")
        
        # Calculate global limits for consistent color scales within this season
        print(f"Calculating global percentile values for {season_name}...")
        global_values = {var: [] for var in VARIABLES}
        
        # Collect seasonal means for percentile calculation
        for variable in VARIABLES:
            # Collect anomalies for this season
            seasonal_anomalies = []
            
            for month_num in month_list:
                month_str = f"{month_num:03d}"
                
                # Load climatology and anomaly data
                clim = load_single_month_climatology(variable, month_str)
                anom = load_single_month_anomaly(variable, year, month_str)
                
                if clim is None or anom is None:
                    continue
                
                # Apply unit conversion
                conversion_factor = UNIT_CONVERSION[variable]
                if conversion_factor != 1.0:
                    anom = anom * conversion_factor
                
                seasonal_anomalies.append(anom)
            
            # Calculate seasonal mean if we have data
            if seasonal_anomalies:
                try:
                    seasonal_stack = xr.concat(seasonal_anomalies, dim='month')
                    seasonal_mean = seasonal_stack.mean(dim='month', skipna=True)
                    
                    mean_vals = seasonal_mean.values
                    valid_vals = mean_vals[~np.isnan(mean_vals)]
                    
                    if len(valid_vals) > 0:
                        global_values[variable].extend(valid_vals.flatten())
                except Exception as e:
                    print(f"  Error calculating mean for {variable} {season_name} {year}: {e}")
        
        # Calculate percentile-based limits
        global_limits = {}
        for variable in VARIABLES:
            if len(global_values[variable]) > 0:
                abs_vals = np.abs(global_values[variable])
                vmax_a = np.percentile(abs_vals, 99)
                global_limits[variable] = [-vmax_a, vmax_a]
                print(f"  {variable}: Using 99th percentile = {vmax_a:.6f}")
            else:
                global_limits[variable] = [-VLIMS[variable]['anom'][1], VLIMS[variable]['anom'][1]]
                print(f"  Warning: Using default limits for {variable}")
        
        # Create nested gridspec for this season's row (1 row × 3 columns for variables)
        season_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=main_gs[s_idx], wspace=0.1)
        
        # Process each variable (column)
        for v_idx, variable in enumerate(VARIABLES):
            print(f"  Processing {season_name} {year} - {variable}...")
            
            # Create axis with proper projection
            ax = fig.add_subplot(season_gs[v_idx], projection=ccrs.PlateCarree(central_longitude=0))
            
            # Set the map extent
            ax.set_extent([lon_min_180, lon_max_180, region['lat_min'], region['lat_max']], 
                         crs=ccrs.PlateCarree())
            
            # Add coastlines and land features
            ax.coastlines(resolution='50m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            
            # Only show left labels on leftmost column
            if v_idx > 0:
                gl.left_labels = False
            
            gl.xlabel_style = {'size': 8, 'color': 'black'}
            gl.ylabel_style = {'size': 8, 'color': 'black'}
            
            # Add subplot title with ONLY variable name (no season name)
            var_name = var_display_names.get(variable, TITLES[variable])
            ax.set_title(f"{var_name} anomalies in {season_name}", fontsize=16)
            
            # Collect anomalies for this season
            seasonal_anomalies = []
            clim = None
            
            for month_num in month_list:
                month_str = f"{month_num:03d}"
                
                # Load climatology and anomaly data
                current_clim = load_single_month_climatology(variable, month_str)
                anom = load_single_month_anomaly(variable, year, month_str)
                
                if current_clim is None or anom is None:
                    continue
                
                # Apply unit conversion
                conversion_factor = UNIT_CONVERSION[variable]
                if conversion_factor != 1.0:
                    current_clim = current_clim * conversion_factor
                    anom = anom * conversion_factor
                
                # Save climatology for coordinate information
                if clim is None:
                    clim = current_clim
                
                seasonal_anomalies.append(anom)
            
            # Skip if we don't have any data for this season
            if len(seasonal_anomalies) == 0 or clim is None:
                print(f"  Error: Insufficient data for {variable}, {season_name} {year}")
                ax.text(0.5, 0.5, f"No data available", 
                      transform=ax.transAxes, ha='center', fontsize=12)
                continue
            
            try:
                # Calculate seasonal mean
                seasonal_stack = xr.concat(seasonal_anomalies, dim='month')
                seasonal_mean = seasonal_stack.mean(dim='month', skipna=True)
                
                # Get limits for consistent coloring
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
                    
                # Create discrete colormap
                discrete_anom_cmap = create_symmetric_discrete_colormap(anom_cmap_base, levels=9)
                
                # Create boundary norm
                levels = 9
                anom_bounds = np.linspace(vmin_a, vmax_a, levels + 1)
                anom_norm = BoundaryNorm(anom_bounds, discrete_anom_cmap.N)
                
                # Plot data
                seasonal_mean_vals = seasonal_mean.compute().values
                
                if use_2d:
                    plot_lon = lon.values - 360 if np.any(lon.values > 180) else lon.values
                    pc = ax.pcolormesh(plot_lon, lat, seasonal_mean_vals, cmap=discrete_anom_cmap, 
                                     norm=anom_norm, transform=ccrs.PlateCarree())
                else:
                    pc = ax.pcolormesh(seasonal_mean_vals, cmap=discrete_anom_cmap, norm=anom_norm,
                                     transform=ccrs.PlateCarree())
                
                # Add colorbar
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
                
                # Create colorbar
                cbar = fig.colorbar(pc, cax=cax, ticks=anom_bounds)
                cbar.set_label(f"Δ {TITLES_UNITS[variable]}", size=10)
                
                # Format tick labels
                if variable == 'temp':
                    cbar.ax.set_yticklabels([f"{t:.1f}" for t in anom_bounds], fontsize=8)
                elif variable == 'TOT_PROD':
                    cbar.ax.set_yticklabels([f"{t:.2f}" for t in anom_bounds], fontsize=8)
                else:
                    cbar.ax.set_yticklabels([f"{t:.1f}" for t in anom_bounds], fontsize=8)
                
            except Exception as e:
                print(f"  Error plotting {variable}, {season_name} {year}: {e}")
                ax.text(0.5, 0.5, f"Error: {str(e)[:50]}", 
                      transform=ax.transAxes, ha='center', fontsize=10)
    
    # Save the entire figure as a single file
    output_file = os.path.join(OUTPUT_DIR, f'seasonal_anomalies_{year}_combined.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {output_file}")
    plt.close(fig)

#--------- MAIN EXECUTION ---------#
if __name__ == "__main__":
    print(f"\n==== Generating combined seasonal anomaly plot for 2014 ====")
    plot_seasonal_2014_anomalies()
