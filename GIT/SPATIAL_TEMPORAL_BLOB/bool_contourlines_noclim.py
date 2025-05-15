"""
For normal operation (process + plot):
cd /home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/ python bool_contourlines.py

To process data only (save intermediate results):
cd /home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/  python bool_contourlines.py --process-only

To plot from saved data (skip processing):
cd /home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/  python bool_contourlines.py --process-only

To specify a custom output filename:
cd /home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/  python bool_contourlines.py --plot-only

python bool_contourlines.py --output


"""
#%% 
#--------- IMPORT REQUIRED MODULES ---------#
import xarray as xr
import numpy as np

import cmocean as cmo
import os
import glob
import re
import pandas as pd
from pathlib import Path
import sys
sys.path.append('/home/bblaser/scripts/bsc_beni/ALL_MODULES')
from get_study_regions import GetRegions
from get_model_datasets_Hplus_surf_only import ModelGetter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

#%% 
#--------- CONFIGURATION ---------#
DATA_DIR = '/nfs/sea/work/bblaser/z_avg_meanpool/'
FILE_PATTERN = 'z_avg_*_*_37zlevs_full_1x1meanpool_downsampling.nc'
VARIABLE_NAME = 'temp'  # Changed from TOT_PROD to temperature
# Extend depth levels to cover 0-100m
DEPTH_LEVELS = [0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60, -65, -70, -75, -80, -85, -90, -95, -100]
OUTPUT_DIR = '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events'
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# File path for the extreme event boolean dataset
BOOLEAN_FILE_PATH = '/nfs/sea/work/bblaser/data/temp/boolean_array_hobday2016_threshold_and_climatology_temp_37zlevs_full_1x1meanpool_downsampling_2011-2021analysisperiod_80perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing.nc'
BOOL_VAR_NAME = 'boolean_smoothed_hobday2016'

# File path for climatologies
CLIM_FILE_PATH = '/nfs/sea/work/bblaser/monthly_clim/'
CLIM_FILE_PATTERN = 'climatology_month_*_2010-2021.nc'

# Redefine the seasons
SUMMER_MONTHS = [4, 5, 6, 7, 8, 9]  # April through September
WINTER_MONTHS = [10, 11, 12, 1, 2, 3]  # October through March

# Target years to analyze
TARGET_YEARS = [2014, 2015, 2016]

# Define a consistent decimation factor as a global constant at the top of the file
DECIMATION_FACTOR = 7  # Choose appropriate value based on your data density

#%%
def load_boolean_data():
    """Load the boolean dataset with extreme events."""
    print("\nLoading boolean extreme event dataset...")
    try:
        bool_ds = xr.open_dataset(BOOLEAN_FILE_PATH)
        if BOOL_VAR_NAME not in bool_ds:
            print(f"ERROR: Variable {BOOL_VAR_NAME} not found in boolean dataset")
            print(f"Available variables: {list(bool_ds.data_vars)}")
            return None
        
        # Select the surface (first depth level)
        bool_surface = bool_ds[BOOL_VAR_NAME].isel(depth=0)
        
        # Get the coordinates
        lons = bool_ds['lon'].values if 'lon' in bool_ds else None
        lats = bool_ds['lat'].values if 'lat' in bool_ds else None
        
        if lons is None or lats is None:
            print("WARNING: Geographic coordinates not found in boolean dataset")
            return None
            
        print(f"✓ Loaded boolean data with dimensions {bool_surface.shape}")
        return bool_ds, bool_surface, lons, lats
        
    except Exception as e:
        print(f"Error loading boolean data: {e}")
        return None

def get_seasonal_extreme_data(bool_ds, year, season_months=None, depth_idx=0):
    """Extract seasonal extreme event frequency for a specific season of a specific year and depth."""
    if season_months is None:
        season_months = SUMMER_MONTHS
        
    if 'time' not in bool_ds.dims:
        print("ERROR: No time dimension in boolean dataset")
        return None
    
    # Get the time values
    doy_values = bool_ds['time'].values
    
    # Storage for monthly data
    monthly_counts = []
    total_days = 0
    
    for month in season_months:
        # Determine correct year for this month in winter season
        current_year = year
        if month in [1, 2, 3] and set(season_months) == set(WINTER_MONTHS):
            current_year = year + 1
            
        # Calculate the DOY range for this month
        doy_start, doy_end = get_month_doy_range(current_year, month)
        
        # Adjust for leap years
        if (current_year % 4 == 0 and current_year % 100 != 0) or current_year % 400 == 0:  # Leap year
            if month > 2:  # After February
                doy_start += 1
                doy_end += 1
        
        # Calculate the time index for this year
        year_offset = (current_year - 2011) * 365  # 2011 is the start year from filename
        if current_year > 2012:  # Add leap days after 2012
            year_offset += (current_year - 2012) // 4
        
        # Calculate indices for this month in this year
        month_indices = np.where((doy_values >= year_offset + doy_start) & 
                               (doy_values <= year_offset + doy_end))
        
        if len(month_indices[0]) == 0:
            print(f"No boolean data found for {month}/{current_year}")
            continue
        
        # Get data for this month at the specified depth
        monthly_data = bool_ds[BOOL_VAR_NAME].isel(depth=depth_idx, time=month_indices[0])
        monthly_counts.append(monthly_data)
        total_days += len(month_indices[0])
        
    # If we couldn't find data for any month, return None
    if not monthly_counts:
        return None
    
    # Combine all monthly data and calculate MEAN across the entire season
    seasonal_data = xr.concat(monthly_counts, dim='time')
    seasonal_frequency = seasonal_data.mean(dim='time')
    
    return seasonal_frequency

def get_month_doy_range(year, month):
    """Get the day of year range for a specific month in a specific year."""
    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days_per_month = [31, 29 if is_leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    doy_start = sum(days_per_month[:month-1]) + 1
    doy_end = doy_start + days_per_month[month-1] - 1
    return (doy_start, doy_end)

def load_files_for_season(year, months):
    """Load files for a specific season of a year."""
    # Debug print statements to check loading process
    print("\n=== DEBUG: SEASON DATA LOADING ===")
    print(f"Initial year for season: {year}")
    print(f"Season months: {months}")
    if set(months) == set(WINTER_MONTHS):
        print(f"Winter season detected - Jan/Feb/Mar files should be from year {year+1}")
    else:
        print(f"Summer season detected - all files should be from year {year}")
    print("===================================\n")
    
    datasets = []
    
    for month in months:
        # Format month with leading zeros (e.g., month 4 = "004")
        month_str = f"{month:03d}"
        
        # Determine the correct year for this month
        file_year = year
        if month in [1, 2, 3] and set(months) == set(WINTER_MONTHS):
            # If this is part of winter season and months are Jan-Mar, use next year
            file_year = year + 1
            
        # Create the file pattern for this month
        file_pattern = os.path.join(DATA_DIR, f"z_avg_{file_year}_{month_str}_37zlevs_full_1x1meanpool_downsampling.nc")
        
        print(f"Looking for file matching: {file_pattern}")
        matching_files = glob.glob(file_pattern)
        
        if not matching_files:
            print(f"No files found for year {file_year}, month {month}")
            continue
        
        print(f"Found {len(matching_files)} file(s) for year {file_year}, month {month}")
        
        # Open the file as a dataset
        try:
            ds = xr.open_dataset(matching_files[0])
            
            # Ensure coordinates are preserved
            if 'lon_rho' not in ds and 'lat_rho' not in ds:
                print("Warning: Geographic coordinates missing in dataset")
            
            datasets.append(ds)
        except Exception as e:
            print(f"Error opening file for {file_year}/{month}: {e}")
    
    if not datasets:
        print(f"No data found for season {year}")
        return None
    
    # Combine all monthly datasets
    return xr.concat(datasets, dim='time')

def calculate_depth_average(ds, variable_name, depth_levels, period_name):
    """Calculate average over specific depth levels with guaranteed coordinate preservation."""
    print(f"\nCalculating {period_name} average for {variable_name}...")
    
    # Make sure our variable exists
    if variable_name not in ds:
        raise ValueError(f"Variable {variable_name} not found in dataset")
        
    # Find the depth dimension
    depth_dim = None
    for dim in ds[variable_name].dims:
        if dim not in ['time', 'xi_rho', 'eta_rho'] and dim in ds.coords:
            depth_dim = dim
            break
    
    if not depth_dim:
        print("WARNING: Could not identify depth dimension by name")
        # Use first non-time, non-spatial dimension
        for dim in ds[variable_name].dims:
            if dim != 'time' and dim not in ['xi_rho', 'eta_rho']:
                depth_dim = dim
                print(f"Using '{depth_dim}' as depth dimension")
                break
    
    if not depth_dim:
        raise ValueError("Could not identify any depth dimension")
    
    # Get units if available
    units = ds[variable_name].attrs.get('units', '')
    
    # Select the requested depth levels
    depth_slices = []
    
    # Handle whether depth dimension has values or just indices
    if hasattr(ds[depth_dim], 'values') and len(ds[depth_dim].values) > 0:
        # Use actual depth values
        for depth in depth_levels:
            slice_data = ds[variable_name].sel({depth_dim: depth}, method='nearest')
            depth_slices.append(slice_data)
            print(f"Selected depth near {depth}m")
    else:
        # Just use the first N levels
        n_levels = min(len(depth_levels), len(ds[depth_dim]))
        print(f"Using first {n_levels} depth levels")
        for i in range(n_levels):
            slice_data = ds[variable_name].isel({depth_dim: i})
            depth_slices.append(slice_data)
    
    # Combine all depth slices
    if depth_slices:
        # Stack along a new dimension
        stacked = xr.concat(depth_slices, dim='z_levels')
        # Calculate vertical average
        vertical_mean = stacked.mean(dim='z_levels', skipna=True)
    else:
        raise ValueError("No valid depth levels found")
    
    # Calculate time average ONLY if time dimension exists
    if 'time' in vertical_mean.dims:
        result = vertical_mean.mean(dim='time', skipna=True)
    else:
        print(f"No time dimension found in {period_name} data, skipping time averaging")
        result = vertical_mean
    
    # Give it a name
    result.name = f"{variable_name}_{period_name}_mean"
    
    # EXPLICIT COORDINATE PRESERVATION
    # Make sure the coordinates are preserved in the result
    coords = {}
    vars_to_add = {}
    
    # Get the dimensions of our result so we can ensure compatibility
    result_dims = set(result.dims)
    print(f"Result dimensions: {result_dims}")
    
    # First, check if coordinates already exist (they should)
    for coord in ['lon_rho', 'lat_rho']:
        coord_data = None
        
        # Try to get from coords or data_vars
        if coord in ds.coords:
            coord_data = ds.coords[coord]
        elif coord in ds.data_vars:
            coord_data = ds[coord]
            
        if coord_data is not None:
            # Check if dimensions are compatible
            extra_dims = set(coord_data.dims) - result_dims
            if extra_dims:
                # Select a specific slice to remove extra dimensions
                print(f"Removing extra dimensions {extra_dims} from {coord}")
                selection = {dim: 0 for dim in extra_dims}
                coord_data = coord_data.isel(**selection)
                
            # Now the coordinate should be compatible
            coords[coord] = coord_data
            vars_to_add[coord] = coord_data
    
    # Apply these coordinates to our result
    if coords:
        print(f"Preserving {len(coords)} geographic coordinates with dimensions: {[c.dims for c in coords.values()]}")
        # First add as coordinates
        result = result.assign_coords(coords)
        
        # Convert to dataset to add variables
        result_ds = result.to_dataset()
        # Add coordinates as data variables
        for name, var in vars_to_add.items():
            result_ds[name] = var
            
        return result_ds, units
    else:
        print("WARNING: No coordinates found to preserve!")
        return result.to_dataset(), units  # Convert to dataset for consistency

def plot_climatology_seasons():
    """Create climatology plots for summer and winter seasons."""
    seasons = [
        {"name": "Summer", "months": SUMMER_MONTHS, "months_text": "Apr-Sep"},
        {"name": "Winter", "months": WINTER_MONTHS, "months_text": "Oct-Mar"}
    ]
    
    # Store results for returning
    results = []
    
    for season in seasons:
        print(f"\nGenerating climatology for {season['name']} ({season['months_text']}) mean...")
        
        # Define the season months with correct formatting
        month_strs = [f"{month:03d}" for month in season['months']]
        print(f"Looking for climatology files with months: {month_strs}")
        
        # Storage for monthly data arrays
        month_data_arrays = []
        units = ""
        lon_data = None
        lat_data = None
        
        # Load each month's climatology directly
        for month_str in month_strs:
            clim_file = os.path.join(CLIM_FILE_PATH, f"climatology_month_{month_str}_2010-2021.nc")
            
            if not os.path.exists(clim_file):
                print(f"Warning: Climatology file not found: {clim_file}")
                continue
                
            print(f"Loading climatology file: {os.path.basename(clim_file)}")
            
            try:
                clim_ds = xr.open_dataset(clim_file)
                
                if VARIABLE_NAME not in clim_ds:
                    print(f"Warning: Variable {VARIABLE_NAME} not found in climatology file")
                    print(f"Available variables: {list(clim_ds.data_vars)}")
                    continue
                
                # Find depth dimension if it exists
                depth_dim = None
                for dim in clim_ds[VARIABLE_NAME].dims:
                    if dim not in ['xi_rho', 'eta_rho', 'time']:
                        depth_dim = dim
                        print(f"Found depth dimension: {depth_dim}")
                        break
                
                # Average across depth levels if depth dimension exists
                if depth_dim and len(DEPTH_LEVELS) > 0:
                    depth_slices = []
                    
                    # Try to select by actual depth values first
                    try:
                        for depth in DEPTH_LEVELS:
                            slice_data = clim_ds[VARIABLE_NAME].sel({depth_dim: depth}, method='nearest')
                            depth_slices.append(slice_data)
                            print(f"Selected depth near {depth}m")
                    except:
                        # Fall back to using indices
                        print("Falling back to depth indices")
                        for i in range(min(DEPTH_LEVELS), len(clim_ds[depth_dim])):
                            slice_data = clim_ds[VARIABLE_NAME].isel({depth_dim: i})
                            depth_slices.append(slice_data)
                            print(f"Using depth index {i}")
                    
                    if depth_slices:
                        depth_mean = xr.concat(depth_slices, dim='z_levels').mean(dim='z_levels', skipna=True)
                        month_data_arrays.append(depth_mean)
                    else:
                        print(f"Warning: No valid depth selections for {month_str}")
                else:
                    # No depth dimension, use the data as is
                    print(f"No depth averaging needed for {month_str}")
                    month_data_arrays.append(clim_ds[VARIABLE_NAME])
                
                # Get units if not already set
                if not units and hasattr(clim_ds[VARIABLE_NAME], 'attrs'):
                    units = clim_ds[VARIABLE_NAME].attrs.get('units', '')
                
                # Store coordinates from first valid file
                if lon_data is None:
                    if 'lon_rho' in clim_ds:
                        lon_data = clim_ds['lon_rho'].values
                    elif 'lon' in clim_ds:
                        lon_data = clim_ds['lon'].values
                        
                if lat_data is None:
                    if 'lat_rho' in clim_ds:
                        lat_data = clim_ds['lat_rho'].values
                    elif 'lat' in clim_ds:
                        lat_data = clim_ds['lat'].values
                    
            except Exception as e:
                print(f"Error processing climatology file {clim_file}: {e}")
        
        if not month_data_arrays:
            print(f"Error: No climatology data could be processed for {season['name']}")
            continue
        
        # Calculate seasonal mean by averaging the monthly climatologies
        print(f"Calculating seasonal mean from {len(month_data_arrays)} monthly climatologies")
        seasonal_mean = xr.concat(month_data_arrays, dim='month').mean(dim='month')
        
        # Store results for returning
        results.append({
            'season': season['name'].lower(),
            'seasonal_mean': seasonal_mean,
            'lon_data': lon_data,
            'lat_data': lat_data,
            'units': units
        })
        
    return results

def plot_seasonal_temp_with_mhw_overlay(ds, seasonal_count, lons, lats, title, year, season):
    """Plot temperature means with marine heatwave frequency overlay."""
    
    # Extract the temperature variable name
    temp_var_name = None
    for var_name in ds.data_vars:
        if var_name not in ['lon_rho', 'lat_rho'] and VARIABLE_NAME in var_name:
            temp_var_name = var_name
            break
    
    if not temp_var_name:
        print("ERROR: No temperature variable found in dataset")
        return None
    
    # Extract data and coordinates
    temp_data = ds[temp_var_name].values
    
    lon_data = None
    lat_data = None
    
    if 'lon_rho' in ds.coords:
        lon_data = ds.coords['lon_rho'].values
    elif 'lon_rho' in ds.data_vars:
        lon_data = ds['lon_rho'].values
        
    if 'lat_rho' in ds.coords:
        lat_data = ds.coords['lat_rho'].values
    elif 'lat_rho' in ds.data_vars:
        lat_data = ds['lat_rho'].values
    
    # Create result dict
    result = {
        'year': year,
        'season': season,
        'temperature_data': temp_data,
        'lon_data': lon_data,
        'lat_data': lat_data,
        'seasonal_count': seasonal_count,
        'mhw_lons': lons,
        'mhw_lats': lats,
        'title': title
    }
    
    # Decimate data - keep only every Nth point (with N=3, 5, 10, etc.)
    result['decimated_count'] = result['seasonal_count'][::DECIMATION_FACTOR, ::DECIMATION_FACTOR] 
    result['decimated_lons'] = result['mhw_lons'][::DECIMATION_FACTOR, ::DECIMATION_FACTOR]
    result['decimated_lats'] = result['mhw_lats'][::DECIMATION_FACTOR, ::DECIMATION_FACTOR]
    
    return result

def create_subplot_grid(climatology_results, seasonal_results, output_file=None):
    """Create a 2x3 subplot grid with MHW overlay data and water column fraction using GridSpec."""
    # Create figure
    fig = plt.figure(figsize=(18, 12), dpi=300)
    
    # Create grid layout with 2 rows and 4 columns (3 for plots, 1 for colorbar)
    gs = gridspec.GridSpec(
        nrows=2, ncols=4, 
        width_ratios=[1, 1, 1, 0.05],  # 3 plot columns + 1 colorbar column
        height_ratios=[1, 1],
        wspace=0.1, hspace=0.25,
        figure=fig
    )
    
    # Create all subplots with cartopy projection
    axes = {}
    for i in range(2):
        for j in range(3):
            axes[(i,j)] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
            axes[(i,j)].set_aspect('auto')
            axes[(i,j)].set_box_aspect(1)
    
    # Add colorbar axis - only one for water column fraction
    cbar_ax = fig.add_subplot(gs[:, 3])  # Water column fraction colorbar
    
    # Map settings - Pacific-centered view
    lon_min, lon_max = 205, 245
    lat_min, lat_max = 20, 62
    
    # Set common map features for all plots
    for i in range(2):
        for j in range(3):
            ax = axes[(i, j)]
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.coastlines(resolution='10m', color='black', linewidth=0.2)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            
            # Add gridlines with 10-degree x ticks
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, 
                             color='gray', alpha=0.5, linestyle='--')
            gl.xlocator = mticker.MultipleLocator(10)
            gl.ylocator = mticker.MultipleLocator(10)
            gl.top_labels = False
            gl.right_labels = False
            
            # Remove this conditional to show x labels on all rows
    
    # Sort results by year
    sorted_results = sorted(seasonal_results, key=lambda x: x['year'])
    
    # Group results by year and season
    results_by_year_season = {}
    for result in sorted_results:
        key = (result['year'], result['season'])
        results_by_year_season[key] = result
    
    # Define the MHW frequency thresholds and visualization
    mhw_levels = [0.0, 0.4, 0.8, 1.0] #0.7, 1.0]
    mhw_colors = [(0,0,0,0), (0,0,0,0.05), (0,0,0,0.1), (0,0,0,0.3)]
    # Define colors for contour lines - using blue for lowest level
    mhw_line_colors = ['red', 'black', 'black', 'black']  # First color is for the 0.1 contour
    mhw_linewidths = [2, 0.5, 0.5, 0.5]  # Consistent, thin lines for all contours
    
    # Define discrete levels for water column fraction - START AT 0.1 instead of 0
    fraction_levels = np.linspace(0.1, 1, 10)  # Start at 0.1 to cut out the black part

    # Create a discrete colormap using the amp colormap, but truncate the lower range
    # This skips the darkest colors at the beginning of the colormap
    cmap = plt.cm.get_cmap(cmo.cm.amp, 9)  # 9 distinct colors
    cmap = truncate_colormap(cmap, 0, 0.7)  # Cut out the dark end of the colormap

    # Create a normalization to map values to discrete colormap indices
    norm = plt.matplotlib.colors.BoundaryNorm(fraction_levels, cmap.N)
    
    # Define plot order: summer at top row, winter at bottom row
    # With columns for years 2014, 2015, 2016
    seasons = ["summer", "winter"]
    
    # Create mapping from year/season to plot position
    for i, season in enumerate(seasons):
        for j, year in enumerate(TARGET_YEARS):
            # Get the result for this year and season
            key = (year, season)
            if key not in results_by_year_season:
                continue
                
            result = results_by_year_season[key]
            ax = axes[(i, j)]
            
            # Get the season months
            season_months = SUMMER_MONTHS if season == "summer" else WINTER_MONTHS
            
            # Calculate the water column fraction for this year and season
            # Load the boolean dataset if not already loaded
            bool_result = load_boolean_data()
            if bool_result is not None:
                bool_ds, _, lons, lats = bool_result
                
                # Calculate water column fraction with fixed approach
                water_column_fraction = calculate_water_column_fraction_fixed(bool_ds, year, season_months)
                
                # Plot water column fraction as background using discrete colormap
                if water_column_fraction is not None:
                    fraction_plot = ax.pcolormesh(
                        lons, lats, water_column_fraction,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        norm=norm
                    )
                                            
            # Add MHW frequency contours with fills
            if result['seasonal_count'] is not None:
                # Check if decimated data exists, if not create it
                if 'decimated_count' not in result and 'seasonal_count' in result:
                    result['decimated_count'] = result['seasonal_count'][::DECIMATION_FACTOR, ::DECIMATION_FACTOR]
                if 'decimated_lons' not in result and 'mhw_lons' in result:
                    result['decimated_lons'] = result['mhw_lons'][::DECIMATION_FACTOR, ::DECIMATION_FACTOR]
                if 'decimated_lats' not in result and 'mhw_lats' in result:
                    result['decimated_lats'] = result['mhw_lats'][::DECIMATION_FACTOR, ::DECIMATION_FACTOR]
                
                # Add filled contours first using the existing color scheme
                cf = ax.contourf(
                    result['decimated_lons'], result['decimated_lats'], 
                    result['decimated_count'], levels=mhw_levels,
                    colors=mhw_colors,  # Use the existing colors
                    transform=ccrs.PlateCarree(),
                    extend='both',
                    antialiased=True  # Enable antialiasing for smoother appearance
                )
                
                # Then add contour lines on top
                cs = ax.contour(
                    result['decimated_lons'], result['decimated_lats'], 
                    result['decimated_count'], levels=mhw_levels[1:],
                    colors=mhw_line_colors, linewidths=mhw_linewidths,
                    transform=ccrs.PlateCarree(),
                    antialiased=True
                )
            
            # Set title
            season_name = "Summer" if season == "summer" else "Winter"
            months_text = "Apr-Sep" if season == "summer" else "Oct-Mar"
            ax.set_title(f"{year} {season_name} ({months_text})")
    
    # Colorbar for water column fraction - with discrete levels
    cbar = fig.colorbar(
        ScalarMappable(cmap=cmap, norm=norm),
        cax=cbar_ax,
        label='Fraction of 0-100m Water Column affected',
        ticks=fraction_levels  # Use the same boundaries for ticks
    )
    
    # Create percentage labels for colorbar
    percentage_labels = [f'{int(level*100)}%' for level in fraction_levels]
    cbar.set_ticklabels(percentage_labels)
    
    # Add legend for MHW frequency contours
    # Generate legend patches automatically based on mhw_levels
    legend_patches = []

    for i in range(len(mhw_levels) - 1):
        start_level = mhw_levels[i]
        end_level = mhw_levels[i+1]
        label = f"{start_level}-{end_level}"
        
        if i == 0:
            # Skip the first level entirely (no patch, no label)
            continue
        
        # For other levels, create a black patch with appropriate alpha
        patch = mpatches.Patch(facecolor=mhw_colors[i], label=label)
        
        # Set edge color and linewidth
        if i == 1:
            patch.set_edgecolor('red')
            patch.set_linewidth(2)
        else:
            patch.set_edgecolor('black')
            patch.set_linewidth(0.5)
        
        legend_patches.append(patch)
 
    # Create legend
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.025),
        title='Fraction of Extreme Days per Season',
        ncol=4,
        frameon=True,
        fancybox=True
    )
    
    # Add title
    fig.suptitle('Marine Heatwave Distribution (2014-2016)\n(Contours: Fraction Extreme Temperature Days per Season | Colors: Fraction of Water Column affected by Extreme Temperature)', 
                fontsize=16, y=0.95)
    
    # Layout
    plt.tight_layout(rect=[0, 0.05, 0.95, 0.93])
    
    # Save figure if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_file}")
    
    plt.show()
    return fig

def calculate_water_column_fraction_fixed(bool_ds, year, season_months):
    """Calculate fraction of water column (0-100m) affected by heatwaves, with improved calculation."""
    print(f"\nCalculating water column fraction for {year}...")
    
    # Get the depth values
    depth_values = bool_ds['depth'].values
    depths_to_use = []
    depth_indices = []
    
    # Find depths within 0-100m range
    for i, depth in enumerate(depth_values):
        abs_depth = abs(float(depth))  # Ensure we have numerical values
        if abs_depth <= 100:
            depths_to_use.append(abs_depth)
            depth_indices.append(i)
    
    print(f"Using {len(depths_to_use)} depth layers within 0-100m range: {depths_to_use}")
    
    if not depth_indices:
        print("No usable depth layers found in 0-100m range!")
        return None
    
    # Get all time indices for this season in this year
    all_time_indices = []
    for month in season_months:
        # Adjust year for winter season
        current_year = year
        if month in [1, 2, 3] and set(season_months) == set(WINTER_MONTHS):
            current_year = year + 1
            
        # Get time indices for this month
        doy_start, doy_end = get_month_doy_range(current_year, month)
        year_offset = (current_year - 2011) * 365
        if current_year > 2012:
            year_offset += (current_year - 2012) // 4
            
        # Find time indices for this month
        doy_values = bool_ds['time'].values
        month_indices = np.where((doy_values >= year_offset + doy_start) & 
                               (doy_values <= year_offset + doy_end))[0]
        all_time_indices.extend(month_indices)
    
    if not all_time_indices:
        print(f"No time points found for {year} and selected season months!")
        return None
        
    print(f"Found {len(all_time_indices)} time points for selected season")
    
    # Get spatial dimensions directly from the boolean array
    bool_var = bool_ds[BOOL_VAR_NAME]
    spatial_dims = [dim for dim in bool_var.dims if dim not in ['time', 'depth']]
    
    if len(spatial_dims) != 2:
        print(f"Expected 2 spatial dimensions, found {len(spatial_dims)}: {spatial_dims}")
        return None
    
    # Initialize result array - use xarray to handle proper coordinates
    result = None
    
    try:
        # Select the data for all depths and times we need
        selected_data = bool_var.isel(depth=depth_indices, time=all_time_indices)
        
        # Calculate fraction of water column affected over time
        # First, get mean across time (frequency at each depth)
        time_mean = selected_data.mean(dim='time')
        
        # Then average across depths to get fraction of water column
        water_column_fraction = time_mean.mean(dim='depth')
        
        # Create DataArray with proper coordinates
        result = water_column_fraction
        print(f"Successfully calculated water column fraction with shape: {result.shape}")
        
    except Exception as e:
        print(f"Error calculating water column fraction: {e}")
        return None
        
    return result

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncate a colormap by specifying the start and endpoint of the colormap."""
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    new_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#%%
def main():
    """Main execution function."""
    print("=" * 50)
    print("ANALYZING OCEAN TEMPERATURE DATA WITH MARINE HEATWAVE OVERLAY")
    print("=" * 50)
    
    # Load the boolean dataset for extreme events
    bool_result = load_boolean_data()
    if bool_result is None:
        print("ERROR: Could not load boolean extreme event data")
        return

    bool_ds, bool_surface, lons, lats = bool_result
    
    # Get climatology for both seasons
    print("\nCalculating temperature climatologies for Summer and Winter...")
    climatology_results = plot_climatology_seasons()
    
    if not climatology_results:
        print("ERROR: Failed to generate climatology data")
        return
    
    # List to store all seasonal results
    all_seasonal_results = []
    
    # Process each year and season
    for year in TARGET_YEARS:
        for season_type, season_months in [("summer", SUMMER_MONTHS), ("winter", WINTER_MONTHS)]:
            print(f"\nProcessing {year} {season_type.upper()} (months: {season_months})")
            
            # Load data for this year and season
            seasonal_data = load_files_for_season(year, season_months)
            if seasonal_data is None:
                print(f"No data found for {year} {season_type}")
                continue
            
            # Calculate temperature mean for this season
            seasonal_mean, units = calculate_depth_average(seasonal_data, VARIABLE_NAME, DEPTH_LEVELS, f"{season_type}")
            
            # Get MHW frequency data
            seasonal_count = get_seasonal_extreme_data(bool_ds, year, season_months=season_months)
            
            if seasonal_count is None:
                print(f"Warning: No MHW data found for {year} {season_type}")
            
            # Create the plot title
            months_text = "Apr-Sep" if season_type == "summer" else "Oct-Mar"
            title = f"{year} {season_type.capitalize()} ({months_text}) Temperature"
            
            # Create result for this season
            result = plot_seasonal_temp_with_mhw_overlay(
                seasonal_mean, seasonal_count, lons, lats, title, year, season_type)
            
            if result:
                all_seasonal_results.append(result)
    
    # Create the multi-panel plot if we have results
    if all_seasonal_results:
        output_file = os.path.join(OUTPUT_DIR, f"{VARIABLE_NAME}_seasonal_means_with_mhw.png")
        create_subplot_grid(climatology_results, all_seasonal_results, output_file)
    else:
        print("No results to plot")

def save_processed_data(climatology_results, all_seasonal_results, filename='processed_data.pkl'):
    """Save processed data to disk for later use."""
    import pickle
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'climatology_results': climatology_results,
            'all_seasonal_results': all_seasonal_results
        }, f)
    print(f"Processed data saved to {output_path}")

def load_processed_data(filename='processed_data.pkl'):
    """Load processed data from disk."""
    import pickle
    input_path = '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/processed_data_80th.pkl' #os.path.join(OUTPUT_DIR, filename)
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded processed data from {input_path}")
        return data['climatology_results'], data['all_seasonal_results']
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None, None

#%%
# CONFIGURATION - Run this cell first to set up your workflow
RUN_PROCESSING = False  # Set to False to skip data processing and use saved data
RUN_PLOTTING = True    # Set to True to generate plots
CUSTOM_OUTPUT_NAME = f"{VARIABLE_NAME}_seasonal_means_with_mhw.png"  # Custom output filename

# MAIN EXECUTION - Run this cell after configuration
climatology_results = None
all_seasonal_results = []

if RUN_PROCESSING:
    # Processing block - load and process data
    print("=" * 50)
    print("ANALYZING OCEAN TEMPERATURE DATA WITH MARINE HEATWAVE OVERLAY")
    print("=" * 50)
    
    # Load the boolean dataset
    bool_result = load_boolean_data()
    if bool_result is None:
        print("ERROR: Could not load boolean extreme event data")
    else:
        bool_ds, bool_surface, lons, lats = bool_result
        
        # Get climatology for both seasons
        print("\nCalculating temperature climatologies...")
        climatology_results = plot_climatology_seasons()
        
        if climatology_results:
            # Process each year and season - THIS PART WAS MISSING
            all_seasonal_results = []
            
            for year in TARGET_YEARS:
                for season_type, season_months in [("summer", SUMMER_MONTHS), ("winter", WINTER_MONTHS)]:
                    print(f"\nProcessing {year} {season_type.upper()} (months: {season_months})")
                    
                    # Load data for this year and season
                    seasonal_data = load_files_for_season(year, season_months)
                    if seasonal_data is None:
                        print(f"No data found for {year} {season_type}")
                        continue
                    
                    # Calculate temperature mean for this season
                    seasonal_mean, units = calculate_depth_average(seasonal_data, VARIABLE_NAME, DEPTH_LEVELS, f"{season_type}")
                    
                    # Get MHW frequency data
                    seasonal_count = get_seasonal_extreme_data(bool_ds, year, season_months=season_months)
                    
                    if seasonal_count is None:
                        print(f"Warning: No MHW data found for {year} {season_type}")
                    
                    # Create the plot title
                    months_text = "Apr-Sep" if season_type == "summer" else "Oct-Mar"
                    title = f"{year} {season_type.capitalize()} ({months_text}) Temperature"
                    
                    # Create result for this season
                    result = plot_seasonal_temp_with_mhw_overlay(
                        seasonal_mean, seasonal_count, lons, lats, title, year, season_type)
                    
                    if result:
                        all_seasonal_results.append(result)
            
            # Save processed data for future use
            save_processed_data(climatology_results, all_seasonal_results)
else:
    # Load previously processed data
    print("Loading previously processed data...")
    climatology_results, all_seasonal_results = load_processed_data()

# Plot the results if requested
if RUN_PLOTTING and climatology_results and all_seasonal_results:
    print("\nCreating final visualization with all data...")
    output_file = os.path.join(OUTPUT_DIR, CUSTOM_OUTPUT_NAME)
    create_subplot_grid(climatology_results, all_seasonal_results, output_file)
else:
    if not climatology_results or not all_seasonal_results:
        print("\nERROR: No data available for plotting. Check processing results.")
    elif not RUN_PLOTTING:
        print("\nPlotting skipped as requested.")
# %%
