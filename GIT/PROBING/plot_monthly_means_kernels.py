#!/usr/bin/env python3

#%%
import xarray as xr
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter1d
import cmocean.cm as cmo
from matplotlib.gridspec import GridSpec
import re
import traceback
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

# Add this after the import statements and before the UNIT_CONVERSION definition
# Define months to include
ANALYSIS_MONTHS = [5, 6, 7, 8, 9, 10] 
# %%

# Define conversion factors for units
UNIT_CONVERSION = {
    'temp': 1.0,  # No conversion needed
    'TOT_PROD': 31536.0,  # Convert mmol C/m²/day to mol C/m²/year (× 365/1000)
    'POC_FLUX_IN': 31536.0  # Convert mmolC/m²/s to molC/m²/y (× seconds_in_year/1000)
}

# Parameters to adjust layout
figure_width = 30        # Width of figure in inches
figure_height = 10       # Height of figure in inches
combined_width = 24      # Width for combined figure
combined_height = 20     # Height for combined figure
subplot_wspace = 0.2     # Width spacing between subplots
subplot_hspace = 0.3     # Height spacing between subplots (for combined plot)
title_pad = 1            # Padding between title and plot
suptitle_y = 0.93        # Position of figure title
tight_layout_rect = [0, 0, 1, 0.96]  # Rectangle for tight_layout
combined_rect = [0, 0, 1, 0.9]       # Rectangle for combined plot tight_layout

# Convert negative longitudes (-180 to 0) to positive (180 to 360)
def convert_lon_to_360(lon):
    """Convert longitude from -180/180 format to 0-360 format."""
    return lon + 360 if lon < 0 else lon

# Define study regions using the provided coordinates, converting to 0-360° format
regions = {}

# Use the provided STUDY_REGIONS
STUDY_REGIONS = [
    {
        'name': 'lownpp_south',
        'lon_min': -122.7, 'lon_max': -120,
        'lat_min': 33.5, 'lat_max': 35
    },
    {
        'name': 'highnpp_central',
        'lon_min': -127, 'lon_max': -125,
        'lat_min': 39.5, 'lat_max': 41
    },
    {
        'name': 'lownpp_central',
        'lon_min': -126, 'lon_max': -124,
        'lat_min': 35.5, 'lat_max': 37.5
    },
    {
        'name': 'lownpp_north',
        'lon_min': -129, 'lon_max': -127.5,
        'lat_min': 42.15, 'lat_max': 43.25
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
        'name': 'gulf_alaska',
        'lon_min': -154, 'lon_max': -135,
        'lat_min': 55, 'lat_max': 61
    },
    {
        'name': 'highnpp_south',
        'lon_min': -122.75, 'lon_max': -121.5,
        'lat_min': 31.75, 'lat_max': 32.8
    },
]


# Convert regions to 0-360 longitude format and create dictionary
for i, region in enumerate(STUDY_REGIONS):
    region_id = str(i+1)
    regions[region_id] = {
        'name': region['name'],
        'lon_min': convert_lon_to_360(region['lon_min']),
        'lon_max': convert_lon_to_360(region['lon_max']),
        'lat_min': region['lat_min'],
        'lat_max': region['lat_max']
    }

# Define study region
STUDY_REGION = {
    "lon_min": 205, "lon_max": 245,  # Full longitude range
    "lat_min": 20, "lat_max": 62,    # Full latitude range
    "name": "Northeast Pacific"
}

def create_region_map(regions, output_dir):
    """Create a map of the Northeast Pacific with all regions marked as rectangles."""
    print("\nCreating region map...")
    
    # Set up colors for different regions
    region_colors = {
        "CalCS_central": "gold",
        "CalCS_north": "orange",
        "CalCS_south": "darkorange", 
        "Gulf_Alaska": "forestgreen",
        "1": "royalblue",
        "2": "red",
        "3": "purple",
        "4": "green",
        "5": "cyan",
        "6": "magenta"
    }
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set map extent
    ax.set_extent([STUDY_REGION["lon_min"], STUDY_REGION["lon_max"], 
                  STUDY_REGION["lat_min"], STUDY_REGION["lat_max"]], 
                  crs=ccrs.PlateCarree())
    
    # Add natural earth features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # Plot each region as a rectangle
    for region_name, region_info in regions.items():
        color = region_colors.get(region_name, "royalblue")
        
        # Draw rectangle using the region bounds
        lon_min = region_info["lon_min"]
        lon_max = region_info["lon_max"]
        lat_min = region_info["lat_min"]
        lat_max = region_info["lat_max"]
        
        # Calculate width and height
        width = lon_max - lon_min
        height = lat_max - lat_min
        
        # Draw rectangle
        rect = Rectangle((lon_min, lat_min), width, height,
                         edgecolor=color, facecolor='none', 
                         linewidth=2, transform=ccrs.PlateCarree())
        ax.add_patch(rect)
        
        # Add region label at the center of the rectangle
        center_lon = (lon_min + lon_max) / 2
        center_lat = (lat_min + lat_max) / 2
        
        ax.text(center_lon, center_lat, region_info["name"], 
               color='black', fontweight='bold', ha='center',
               transform=ccrs.PlateCarree(), 
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Add title
    ax.set_title(f"Analysis Regions - {STUDY_REGION['name']}", fontsize=14)
    
    # Save figure
    map_output = os.path.join(output_dir, "region_map.png")
    plt.savefig(map_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Region map saved to: {map_output}")


# This should be a separate function
def process_dataset(ds):
    """Process a dataset (potentially applying geographic filters)."""
    print(f"  Processing entire domain (no geographic filtering)")
    return ds

# Update the month extraction in the load_climatology function
# 3. Update the climatology calculation for daily resolution
def load_climatology(clim_files):
    """
    Load daily climatology data and return dictionaries for temperature and productivity.
    """
    clim_temp_by_day = {}
    clim_prod_by_day = {}
    clim_poc_flux_by_day = {}
    
    if not clim_files:
        print(f"Warning: No climatology files found!")
        return clim_temp_by_day, clim_prod_by_day, clim_poc_flux_by_day
    
    print(f"\nLoading daily climatology data...")
    
    for clim_file in clim_files:
        try:
            # Extract month and day from filename (format: climatology_day_MM_DD_2011-2021.nc)
            day_match = re.search(r'day_(\d{2})_(\d{2})_', os.path.basename(clim_file))
            if day_match:
                month = int(day_match.group(1))
                day = int(day_match.group(2))
                
                # Create a day identifier (mmdd format)
                day_id = month * 100 + day
                
                print(f"Processing climatology for {month:02d}-{day:02d}")
                
                # Open the dataset
                ds_clim = xr.open_dataset(clim_file)
                
                # Process dataset (no filtering)
                ds_clim = process_dataset(ds_clim)
                
                # Select the top 50m
                if 'depth' in ds_clim.dims:
                    ds_clim_top = ds_clim.where(ds_clim['depth'] >= -50, drop=True)
                    if ds_clim_top['depth'].size == 0:
                        print(f"  No depth levels within the top 50 m. Skipping.")
                        ds_clim.close()
                        continue
                else:
                    ds_clim_top = ds_clim
                
                # Calculate means for temperature and TOT_PROD
                if 'temp' in ds_clim_top:
                    # Get all spatial dimensions except for depth
                    spatial_dims = [dim for dim in ds_clim_top['temp'].dims]
                    if 'depth' in spatial_dims:
                        spatial_dims.remove('depth')
                    
                    # Calculate mean and safely convert to scalar
                    temp_mean = ds_clim_top['temp'].mean(dim=spatial_dims, skipna=True).values
                    # Use mean() to reduce to a single value if it's still an array
                    if temp_mean.size > 1:
                        temp_value = float(np.mean(temp_mean))
                    else:
                        temp_value = float(temp_mean.item())
                    clim_temp_by_day[day_id] = temp_value
                
                if 'TOT_PROD' in ds_clim_top:
                    # Get all spatial dimensions except for depth
                    spatial_dims = [dim for dim in ds_clim_top['TOT_PROD'].dims]
                    if 'depth' in spatial_dims:
                        spatial_dims.remove('depth')
                        
                    # Calculate mean and safely convert to scalar
                    prod_mean = ds_clim_top['TOT_PROD'].mean(dim=spatial_dims, skipna=True).values
                    # Use mean() to reduce to a single value if it's still an array
                    if prod_mean.size > 1:
                        prod_value = float(np.mean(prod_mean))
                    else:
                        prod_value = float(prod_mean.item())
                    clim_prod_by_day[day_id] = prod_value

                if 'POC_FLUX_IN' in ds_clim:
                    # Find the closest depth to 100m
                    if 'depth' in ds_clim.dims:
                        # Get the index of the closest depth to -100m
                        target_depth = -100
                        depth_idx = np.abs(ds_clim['depth'].values - target_depth).argmin()
                        closest_depth = ds_clim['depth'].values[depth_idx]
                        
                        print(f"  Using depth {closest_depth}m for POC_FLUX_IN (target: -100m)")
                        
                        # Extract POC_FLUX_IN at this depth
                        poc_flux_at_100m = ds_clim['POC_FLUX_IN'].isel(depth=depth_idx)
                        
                        # Calculate mean across all other dimensions
                        spatial_dims = [dim for dim in poc_flux_at_100m.dims]
                        poc_flux_mean = poc_flux_at_100m.mean(dim=spatial_dims, skipna=True).values
                        
                        # Convert to a scalar if needed
                        if poc_flux_mean.size > 1:
                            poc_flux_value = float(np.mean(poc_flux_mean))
                        else:
                            poc_flux_value = float(poc_flux_mean.item())
                            
                        clim_poc_flux_by_day[day_id] = poc_flux_value
                
                ds_clim.close()
            else:
                print(f"  Couldn't extract day information from {clim_file}")
                
        except Exception as e:
            print(f"Error processing climatology file {clim_file}: {e}")
            traceback.print_exc()
    
    print(f"Loaded climatology data for {len(clim_temp_by_day)} days")
    return clim_temp_by_day, clim_prod_by_day, clim_poc_flux_by_day

def process_data_files(nc_files, clim_temp_by_day, clim_prod_by_day, clim_poc_flux_by_day):
    """Process monthly data files with daily timesteps."""
    temp_values = []
    tot_prod_values = []
    poc_flux_values = []
    temp_anomalies = []
    prod_anomalies = []
    poc_flux_anomalies = []
    dates = []
    
    if not nc_files:
        print(f"Warning: No data files found!")
        return pd.DataFrame()
    
    print(f"\nProcessing data files...")
    
    # Sort files chronologically
    nc_files = sorted(nc_files)
    
    for file in nc_files:
        try:
            # Extract year and month from filename
            # Format: z_avg_YYYY_MMM_37zlevs_full_1x1meanpool_downsampling_5x5.nc
            match = re.search(r'z_avg_(\d{4})_(\d{3})_', os.path.basename(file))
            
            if not match:
                print(f"Could not extract date from {file}, skipping...")
                continue
                
            year = int(match.group(1))
            month_str = match.group(2)
            month = int(month_str)

            # Filter out winter months
            if month not in ANALYSIS_MONTHS:
                print(f"Skipping {year}-{month:02d}: Not in analysis window")
                continue
            
            print(f"Processing {year}-{month:02d}: {os.path.basename(file)}")
            
            # Open the dataset
            ds = xr.open_dataset(file)
            ds = process_dataset(ds)
            
            # Check if we have time dimension (daily data within the month)
            if 'time' not in ds.dims:
                print(f"  Warning: No time dimension in {file}, skipping...")
                ds.close()
                continue
            
            # Process each day in the month
            for time_idx in range(ds.dims['time']):
                # Assign day values (1-based) - assuming time index corresponds to day of month
                day = time_idx + 1
                
                # Create date for this day
                file_date = pd.Timestamp(year=year, month=month, day=day)
                day_id = month * 100 + day  # Format as mmdd
                
                print(f"  Processing day {day}: {file_date.strftime('%Y-%m-%d')}")
                
                # Extract data for this day
                ds_day = ds.isel(time=time_idx)
                
                # Select the top 50 m
                if 'depth' in ds_day.dims:
                    ds_top = ds_day.where(ds_day['depth'] >= -50, drop=True)
                    if ds_top['depth'].size == 0:
                        print(f"    No depth levels within the top 50 m. Skipping.")
                        continue
                else:
                    ds_top = ds_day
                
                # Calculate means for temperature and TOT_PROD
                temp_value = np.nan
                prod_value = np.nan
                
                if 'temp' in ds_top:
                    # Average over depth (for top 50m only)
                    if 'depth' in ds_top['temp'].dims:
                        temp_depth_mean = ds_top['temp'].mean(dim='depth', skipna=True)
                    else:
                        temp_depth_mean = ds_top['temp']
                    
                    # Average over remaining spatial dimensions
                    spatial_dims = [dim for dim in temp_depth_mean.dims]
                    temp_value = temp_depth_mean.mean(dim=spatial_dims, skipna=True).values.item()
                else:
                    print(f"    Variable 'temp' not found")
                
                if 'TOT_PROD' in ds_top:
                    # Average over depth (for top 50m only)
                    if 'depth' in ds_top['TOT_PROD'].dims:
                        prod_depth_mean = ds_top['TOT_PROD'].mean(dim='depth', skipna=True)
                    else:
                        prod_depth_mean = ds_top['TOT_PROD']
                    
                    # Average over remaining spatial dimensions
                    spatial_dims = [dim for dim in prod_depth_mean.dims]
                    prod_value = prod_depth_mean.mean(dim=spatial_dims, skipna=True).values.item()
                else:
                    print(f"    Variable 'TOT_PROD' not found")
                
                # Calculate POC flux at 100m
                poc_flux_value = np.nan
                
                if 'POC_FLUX_IN' in ds_day:
                    # Find the closest depth to 100m
                    if 'depth' in ds_day.dims:
                        # Get the index of the closest depth to -100m
                        target_depth = -100
                        depth_idx = np.abs(ds_day['depth'].values - target_depth).argmin()
                        closest_depth = ds_day['depth'].values[depth_idx]
                        
                        # Extract POC_FLUX_IN at this depth
                        poc_flux_at_100m = ds_day['POC_FLUX_IN'].isel(depth=depth_idx)
                        
                        # Average over remaining spatial dimensions
                        spatial_dims = [dim for dim in poc_flux_at_100m.dims]
                        poc_flux_value = poc_flux_at_100m.mean(dim=spatial_dims, skipna=True).values.item()
                    else:
                        print(f"    Variable 'POC_FLUX_IN' found but no depth dimension")
                else:
                    print(f"    Variable 'POC_FLUX_IN' not found")
                
                # Lookup the corresponding climatology value for this day
                clim_temp = clim_temp_by_day.get(day_id, np.nan)
                clim_prod = clim_prod_by_day.get(day_id, np.nan)
                clim_poc_flux = clim_poc_flux_by_day.get(day_id, np.nan)
                
                # Calculate anomalies
                temp_anomaly = temp_value - clim_temp if not np.isnan(temp_value) and not np.isnan(clim_temp) else np.nan
                prod_anomaly = prod_value - clim_prod if not np.isnan(prod_value) and not np.isnan(clim_prod) else np.nan
                poc_flux_anomaly = poc_flux_value - clim_poc_flux if not np.isnan(poc_flux_value) and not np.isnan(clim_poc_flux) else np.nan
                
                # Store values
                dates.append(file_date)
                temp_values.append(temp_value)
                tot_prod_values.append(prod_value)
                poc_flux_values.append(poc_flux_value)
                temp_anomalies.append(temp_anomaly)
                prod_anomalies.append(prod_anomaly)
                poc_flux_anomalies.append(poc_flux_anomaly)
            
            ds.close()
        except Exception as e:
            print(f"Error processing {file}: {e}")
            traceback.print_exc()
            continue
    
    # Convert lists to arrays and create a DataFrame
    if dates:
        df = pd.DataFrame({
            'Date': dates,
            'Temperature': temp_values,
            'TOT_PROD': tot_prod_values,
            'POC_FLUX_IN': poc_flux_values,
            'Temp_Anomaly': temp_anomalies,
            'Prod_Anomaly': prod_anomalies,
            'POC_Flux_Anomaly': poc_flux_anomalies
        })
        
        # Sort by date
        df = df.sort_values('Date')
        return df
    else:
        return pd.DataFrame()

def filter_for_analysis_months(df):
    """Filter a DataFrame to only include rows from the specified analysis months."""
    if 'Date' not in df.columns or df.empty:
        return df
    
    # Apply filter to include only selected months
    return df[df['Date'].dt.month.isin(ANALYSIS_MONTHS)]

# Add this new function after filter_for_analysis_months
def calculate_monthly_means(df):
    """Convert daily data to monthly means."""
    if df.empty:
        return pd.DataFrame()
    
    # Extract year and month as separate columns first
    df_temp = df.copy()
    df_temp['Year'] = df_temp['Date'].dt.year
    df_temp['Month'] = df_temp['Date'].dt.month
    
    # Group by these explicit columns
    monthly_df = df_temp.groupby(['Year', 'Month']).agg({
        'Temperature': 'mean',
        'TOT_PROD': 'mean',
        'POC_FLUX_IN': 'mean',
        'Temp_Anomaly': 'mean',
        'Prod_Anomaly': 'mean', 
        'POC_Flux_Anomaly': 'mean'
    }).reset_index()
    
    # Create proper date column (use 1st day of each month)
    monthly_df['Date'] = pd.to_datetime(
        monthly_df.apply(lambda row: f"{int(row['Year'])}-{int(row['Month'])}-01", axis=1))
    
    return monthly_df

def generate_climatology_series(df, clim_temp_by_day, clim_prod_by_day, clim_poc_flux_by_day):
    """Generate daily climatology data for the full time range of the dataframe."""
    if df.empty:
        return pd.DataFrame()
        
    # Create a list of dates that includes only ANALYSIS_MONTHS for each year in the data range
    start_year = df['Date'].min().year
    end_year = df['Date'].max().year
    
    all_dates = []
    for year in range(start_year, end_year + 1):
        for month in ANALYSIS_MONTHS:
            # Get number of days in the month
            if month in [4, 6, 9, 11]:  # Apr, Jun, Sep, Nov
                days = 30
            elif month == 2:  # Feb
                days = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
            else:
                days = 31
            
            for day in range(1, days + 1):
                date = pd.Timestamp(year=year, month=month, day=day)
                # Check if date is within range of data
                if df['Date'].min() <= date <= df['Date'].max():
                    all_dates.append(date)
    
    date_range = pd.DatetimeIndex(sorted(all_dates))
    
    # Initialize lists to store climatology values
    clim_temp_series = []
    clim_prod_series = []
    clim_poc_flux_series = []
    clim_dates = []
    
    # For each date, look up the appropriate daily climatology value
    for date in date_range:
        day_id = date.month * 100 + date.day  # Format as mmdd
        
        # Only include a date if we have data for that day
        if day_id in clim_temp_by_day or day_id in clim_prod_by_day or day_id in clim_poc_flux_by_day:
            clim_dates.append(date)
            clim_temp_series.append(clim_temp_by_day.get(day_id, np.nan))
            clim_prod_series.append(clim_prod_by_day.get(day_id, np.nan))
            clim_poc_flux_series.append(clim_poc_flux_by_day.get(day_id, np.nan))
    
    # Create climatology dataframe with explicit datetime conversion
    clim_df = pd.DataFrame({
        'Date': pd.to_datetime(clim_dates),  # Explicitly convert to datetime
        'Temp_Climatology': clim_temp_series,
        'Prod_Climatology': clim_prod_series,
        'POC_Flux_Climatology': clim_poc_flux_series
    })
    
    return clim_df

def plot_region_data(ax_compare, ax_anomaly, df, clim_df, region_name, var_type='temp'):
    """Plot monthly comparison and anomaly for a specific variable and region."""
    conversion_factor = 3600*24*365.25/1000
    blob_years = [2014, 2015, 2016]
    
    # Filter data for analysis months only
    df = filter_for_analysis_months(df)
    
    if df.empty:
        ax_compare.text(0.5, 0.5, f"No data for {region_name}", ha='center', transform=ax_compare.transAxes)
        ax_anomaly.text(0.5, 0.5, f"No data for {region_name}", ha='center', transform=ax_anomaly.transAxes)
        return
    
    # Calculate monthly means for plotting
    monthly_df = calculate_monthly_means(df)
    
    # Set up variables based on type
    if var_type == 'temp':
        var_label = 'Temperature'
        clim_var = 'Temp_Climatology'
        anomaly_var = 'Temp_Anomaly'
        color = 'blue'
        unit = '°C'
        convert = False
        y_min, y_max = 0, 25
        anomaly_min, anomaly_max = -3, 3
        # Get monthly climatology values directly
        monthly_clim_values = {month: value for month, value in new_clim_temp_by_month.items() if month in ANALYSIS_MONTHS}
    elif var_type == 'poc':
        var_label = 'POC_FLUX_IN'
        clim_var = 'POC_Flux_Climatology'
        anomaly_var = 'POC_Flux_Anomaly'
        color = 'purple'
        unit = 'mol/m²/year'
        convert = True
        y_min, y_max = 0, 20
        anomaly_min, anomaly_max = -10, 10
        # Get monthly climatology values directly
        monthly_clim_values = {month: value for month, value in new_clim_poc_flux_by_month.items() if month in ANALYSIS_MONTHS}
    else:  # productivity
        var_label = 'TOT_PROD'
        clim_var = 'Prod_Climatology'
        anomaly_var = 'Prod_Anomaly'
        color = 'green'
        unit = 'mol/m³/year'
        convert = True
        y_min, y_max = 0, 1.8
        anomaly_min, anomaly_max = -1.4, 1.4
        # Get monthly climatology values directly
        monthly_clim_values = {month: value for month, value in new_clim_prod_by_month.items() if month in ANALYSIS_MONTHS}
    
    # Create x-coordinate mapping for continuous plotting
    monthly_df = monthly_df.sort_values('Date')
    
    # Define functions to map date to a continuous x-coordinate
    def date_to_x(date):
        """Convert a date to a continuous x-coordinate with no gaps between months."""
        year_idx = date.year - monthly_df['Date'].dt.year.min()
        month_idx = ANALYSIS_MONTHS.index(date.month) if date.month in ANALYSIS_MONTHS else 0
        
        # Final x-coordinate: simple sequential index with no gaps
        x_coord = year_idx * len(ANALYSIS_MONTHS) + month_idx
        return x_coord
    
    # Apply the function to monthly dataframes
    monthly_df['x_coord'] = monthly_df['Date'].apply(date_to_x)
    
    # PLOT DATA VALUES
    # Plot monthly points with larger markers
    if convert:
        ax_compare.scatter(monthly_df['x_coord'], monthly_df[var_label] * conversion_factor, 
                         s=25, color=color, alpha=0.8, marker='o', label='Monthly Means')
        
        # 2. FIX CONTINUOUS LINE - Plot lines by year only
        for year in monthly_df['Date'].dt.year.unique():
            year_data = monthly_df[monthly_df['Date'].dt.year == year]
            ax_compare.plot(year_data['x_coord'], year_data[var_label] * conversion_factor,
                          '-', color=color, alpha=0.5, linewidth=1.0)
    else:
        ax_compare.scatter(monthly_df['x_coord'], monthly_df[var_label], 
                         s=25, color=color, alpha=0.8, marker='o', label='Monthly Means')
        
        # 2. FIX CONTINUOUS LINE - Plot lines by year only
        for year in monthly_df['Date'].dt.year.unique():
            year_data = monthly_df[monthly_df['Date'].dt.year == year]
            ax_compare.plot(year_data['x_coord'], year_data[var_label],
                          '-', color=color, alpha=0.5, linewidth=1.0)
    
    # PLOT CLIMATOLOGY PROPERLY - MONTHLY PATTERN REPEATING FOR EACH YEAR
    years = sorted(monthly_df['Date'].dt.year.unique())
    min_year = min(years)
    
    # Create repeating seasonal climatology pattern for each year
    for year_idx, year in enumerate(years):
        clim_x_coords = []
        clim_y_values = []
        
        # Plot the climatology pattern for each month in this year
        for month_idx, month in enumerate(ANALYSIS_MONTHS):
            if month in monthly_clim_values:
                # X coordinate = year offset + month position
                x_coord = year_idx * len(ANALYSIS_MONTHS) + month_idx
                clim_x_coords.append(x_coord)
                
                # Get the climatology value for this month
                clim_value = monthly_clim_values[month]
                if convert:
                    clim_value *= conversion_factor
                clim_y_values.append(clim_value)
        
        # Plot the seasonal pattern for this year
        if clim_x_coords:
            ax_compare.plot(clim_x_coords, clim_y_values, 'k--', 
                        linewidth=1, alpha=1, 
                        label='Monthly Climatology' if year == years[0] else None)
            
    # Rest of the function continues as before...
    # Highlight blob periods - improved approach
    year_ranges = {}
    
    # First identify the range for each year
    for year in monthly_df['Date'].dt.year.unique():
        year_data = monthly_df[monthly_df['Date'].dt.year == year]
        if not year_data.empty:
            year_ranges[year] = (year_data['x_coord'].min() - 0.1, year_data['x_coord'].max() + 0.1)
    
    # Create blob_periods list to store the ranges for blob years
    blob_periods = []
    
    # Now highlight blob years
    for year in blob_years:
        if year in year_ranges:
            start, end = year_ranges[year]
            ax_compare.axvspan(start, end, color='salmon', alpha=0.2,
                              label='Blob Period' if year == blob_years[0] else None)
            # Store this range for the anomaly plot
            blob_periods.append((start, end))
    
    # Set up x-axis ticks and labels for continuous timeline
    years = sorted(monthly_df['Date'].dt.year.unique())
    tick_positions = []
    tick_labels = []
    
    # Create year labels at the middle of each year's data
    for year in years:
        year_data = monthly_df[monthly_df['Date'].dt.year == year]
        mid_x = (year_data['x_coord'].min() + year_data['x_coord'].max()) / 2
        tick_positions.append(mid_x)
        tick_labels.append(str(year))
    
    ax_compare.set_xticks(tick_positions)
    ax_compare.set_xticklabels(tick_labels)
    
    # Set y-axis label and title
    ax_compare.set_ylabel(f'{var_label} ({unit})')
    if var_type == 'poc':
        ax_compare.set_title(f'{var_label}: {region_name} (100m) - May-Oct Only', pad=title_pad)
    else:
        ax_compare.set_title(f'{var_label}: {region_name} (Top 50m) - May-Oct Only', pad=title_pad)
    
    ax_compare.set_ylim(y_min, y_max)
    ax_compare.legend(loc='best', fontsize='small')
    ax_compare.grid(True, alpha=0.3)
    
    # ANOMALY PLOT
    # Plot anomaly data using monthly means
    if convert:
        ax_anomaly.scatter(monthly_df['x_coord'], monthly_df[anomaly_var] * conversion_factor, 
                         s=25, color=color, alpha=0.8, marker='o', label='Monthly Anomalies')
        
        # Plot lines by year only
        for year in monthly_df['Date'].dt.year.unique():
            year_data = monthly_df[monthly_df['Date'].dt.year == year]
            ax_anomaly.plot(year_data['x_coord'], year_data[anomaly_var] * conversion_factor,
                          '-', color=color, alpha=0.5, linewidth=1.0)
    else:
        ax_anomaly.scatter(monthly_df['x_coord'], monthly_df[anomaly_var], 
                         s=25, color=color, alpha=0.8, marker='o', label='Monthly Anomalies')
        
        # Plot lines by year only
        for year in monthly_df['Date'].dt.year.unique():
            year_data = monthly_df[monthly_df['Date'].dt.year == year]
            ax_anomaly.plot(year_data['x_coord'], year_data[anomaly_var],
                          '-', color=color, alpha=0.5, linewidth=1.0)
    
    # Reference line at zero
    ax_anomaly.axhline(y=0.0, color='black', linestyle='--', linewidth=2, alpha=0.9, label='Climatology Reference')
    
    # Same blob period shading
    for start, end in blob_periods:
        ax_anomaly.axvspan(start, end, color='salmon', alpha=0.2,
                          label='Blob Period' if start == blob_periods[0][0] else None)
    
    min_year = min(years) if years else monthly_df['Date'].dt.year.min()
    ax_anomaly = setup_continuous_xaxis(ax_anomaly, monthly_df, min_year)
    
    ax_anomaly.set_ylabel(f'{var_label} Anomaly ({unit})')
    if var_type == 'poc':
        ax_anomaly.set_title(f'{var_label} Anomaly: {region_name} (100m) - May-Oct Only', pad=title_pad)
    else:
        ax_anomaly.set_title(f'{var_label} Anomaly: {region_name} (Top 50m) - May-Oct Only', pad=title_pad)
    
    ax_anomaly.set_ylim(anomaly_min, anomaly_max)
    ax_anomaly.legend(loc='best', fontsize='small')
    ax_anomaly.grid(True, alpha=0.3)
    
    # Calculate statistics for text box - using original df (daily data)
    pre_blob = df[df['Date'].dt.year < 2014]
    blob = df[df['Date'].dt.year.isin(blob_years)]
    post_blob = df[df['Date'].dt.year > 2016]
    
    pre_blob_mean = pre_blob[anomaly_var].mean() * (conversion_factor if convert else 1)
    blob_mean = blob[anomaly_var].mean() * (conversion_factor if convert else 1)
    post_blob_mean = post_blob[anomaly_var].mean() * (conversion_factor if convert else 1)
    
    # Add text annotations
    stats_text = (
        f"Mean Anomalies:\n"
        f"Pre-Blob (2011-2013): {pre_blob_mean:.8f} {unit}\n"
        f"During Blob (2014-2016): {blob_mean:.8f} {unit}\n"
        f"Post-Blob (2017-2021): {post_blob_mean:.8f} {unit}"
    )
    
    # Add text box
    ax_anomaly.text(0.98, 0.97, stats_text, transform=ax_anomaly.transAxes, 
                  fontsize=9, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

def setup_continuous_xaxis(ax, monthly_df, min_year):
    """Setup x-axis for continuous month display without winter gaps."""
    # Get all years in the data
    years = sorted(monthly_df['Date'].dt.year.unique())
    
    # Create tick positions and labels
    tick_positions = []
    tick_labels = []
    
    for i, year in enumerate(years):
        # Middle position for each season
        mid_season = i * len(ANALYSIS_MONTHS) + len(ANALYSIS_MONTHS) // 2
        tick_positions.append(mid_season)
        tick_labels.append(str(year))
        
        # Add ticks for each month in ANALYSIS_MONTHS
        for j, month in enumerate(ANALYSIS_MONTHS):
            pos = i * len(ANALYSIS_MONTHS) + j
            # Add small ticks for months
            if j > 0:  # Don't add first month to avoid crowding with year label
                ax.axvline(pos, color='gray', linestyle=':', alpha=0.3)
    
    # Set the tick positions and labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # Set x-axis limits slightly beyond the data range
    max_x = max(monthly_df['x_coord']) + 0.5 if not monthly_df.empty else 0.5
    min_x = min(monthly_df['x_coord']) - 0.5 if not monthly_df.empty else -0.5
    ax.set_xlim(min_x, max_x)
    
    # Add month labels below the years
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    # Calculate month tick positions
    month_positions = []
    month_labels = []
    
    for i, month in enumerate(ANALYSIS_MONTHS):
        # Position for each month
        positions = []
        for year_idx, year in enumerate(years):
            positions.append(year_idx * len(ANALYSIS_MONTHS) + i)
        
        # Only use months that actually appear in the data
        if positions:
            month_name = pd.Timestamp(2000, month, 1).strftime('%b')
            month_labels.append(month_name)
            month_positions.append(np.mean(positions))
    
    ax2.set_xticks(month_positions)
    ax2.set_xticklabels(month_labels, fontsize=8)
    ax2.tick_params(axis='x', which='major', pad=0)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 20))
    
    return ax

def create_anomaly_summary(df, region_name, output_dir):
    """Create a summary of anomalies for all variables across different periods."""
    from scipy.stats import spearmanr
    
    blob_start = pd.Timestamp('2014-01-01')
    blob_end = pd.Timestamp('2016-12-31')
    conversion_factor = 3600*24*365.25/1000
    
    # Define periods
    pre_blob = df[df['Date'] < blob_start]
    blob_period = df[(df['Date'] >= blob_start) & (df['Date'] <= blob_end)]
    post_blob = df[df['Date'] > blob_end]
    
    # Calculate period durations in months
    pre_blob_months = len(pre_blob)
    blob_months = len(blob_period)
    post_blob_months = len(post_blob)
    
    # Calculate statistics
    stats = {
        'Temperature': {
            'Pre-Blob': pre_blob['Temp_Anomaly'].mean(),
            'Blob': blob_period['Temp_Anomaly'].mean(),
            'Post-Blob': post_blob['Temp_Anomaly'].mean(),
            'Pre-Blob/Month': pre_blob['Temp_Anomaly'].sum() / pre_blob_months if pre_blob_months > 0 else np.nan,
            'Blob/Month': blob_period['Temp_Anomaly'].sum() / blob_months if blob_months > 0 else np.nan,
            'Post-Blob/Month': post_blob['Temp_Anomaly'].sum() / post_blob_months if post_blob_months > 0 else np.nan,
            'Pre-Blob Total': pre_blob['Temp_Anomaly'].sum(),
            'Blob Total': blob_period['Temp_Anomaly'].sum(),
            'Post-Blob Total': post_blob['Temp_Anomaly'].sum(),
            'Unit': '°C'
        },
        'Productivity': {
            'Pre-Blob': pre_blob['Prod_Anomaly'].mean() * conversion_factor,
            'Blob': blob_period['Prod_Anomaly'].mean() * conversion_factor,
            'Post-Blob': post_blob['Prod_Anomaly'].mean() * conversion_factor,
            'Pre-Blob/Month': pre_blob['Prod_Anomaly'].sum() * conversion_factor / pre_blob_months if pre_blob_months > 0 else np.nan,
            'Blob/Month': blob_period['Prod_Anomaly'].sum() * conversion_factor / blob_months if blob_months > 0 else np.nan,
            'Post-Blob/Month': post_blob['Prod_Anomaly'].sum() * conversion_factor / post_blob_months if post_blob_months > 0 else np.nan,
            'Pre-Blob Total': pre_blob['Prod_Anomaly'].sum() * conversion_factor,
            'Blob Total': blob_period['Prod_Anomaly'].sum() * conversion_factor,
            'Post-Blob Total': post_blob['Prod_Anomaly'].sum() * conversion_factor,
            'Unit': 'mol/m³/year'
        },
        'POC Flux': {
            'Pre-Blob': pre_blob['POC_Flux_Anomaly'].mean() * conversion_factor,
            'Blob': blob_period['POC_Flux_Anomaly'].mean() * conversion_factor,
            'Post-Blob': post_blob['POC_Flux_Anomaly'].mean() * conversion_factor,
            'Pre-Blob/Month': pre_blob['POC_Flux_Anomaly'].sum() * conversion_factor / pre_blob_months if pre_blob_months > 0 else np.nan,
            'Blob/Month': blob_period['POC_Flux_Anomaly'].sum() * conversion_factor / blob_months if blob_months > 0 else np.nan,
            'Post-Blob/Month': post_blob['POC_Flux_Anomaly'].sum() * conversion_factor / post_blob_months if post_blob_months > 0 else np.nan,
            'Pre-Blob Total': pre_blob['POC_Flux_Anomaly'].sum() * conversion_factor,
            'Blob Total': blob_period['POC_Flux_Anomaly'].sum() * conversion_factor,
            'Post-Blob Total': post_blob['POC_Flux_Anomaly'].sum() * conversion_factor,
            'Unit': 'mol/m²/year'
        }
    }
    
    # Calculate Spearman rank correlations
    # Pre-Blob correlations
    if len(pre_blob) > 1:
        temp_prod_corr_pre = spearmanr(pre_blob['Temp_Anomaly'].dropna(), 
                                       pre_blob['Prod_Anomaly'].dropna())[0]
        temp_poc_corr_pre = spearmanr(pre_blob['Temp_Anomaly'].dropna(), 
                                      pre_blob['POC_Flux_Anomaly'].dropna())[0]
    else:
        temp_prod_corr_pre = np.nan
        temp_poc_corr_pre = np.nan
    
    # Blob period correlations
    if len(blob_period) > 1:
        temp_prod_corr_blob = spearmanr(blob_period['Temp_Anomaly'].dropna(), 
                                        blob_period['Prod_Anomaly'].dropna())[0]
        temp_poc_corr_blob = spearmanr(blob_period['Temp_Anomaly'].dropna(), 
                                       blob_period['POC_Flux_Anomaly'].dropna())[0]
    else:
        temp_prod_corr_blob = np.nan
        temp_poc_corr_blob = np.nan
    
    # Post-Blob correlations
    if len(post_blob) > 1:
        temp_prod_corr_post = spearmanr(post_blob['Temp_Anomaly'].dropna(), 
                                        post_blob['Prod_Anomaly'].dropna())[0]
        temp_poc_corr_post = spearmanr(post_blob['Temp_Anomaly'].dropna(), 
                                       post_blob['POC_Flux_Anomaly'].dropna())[0]
    else:
        temp_prod_corr_post = np.nan
        temp_poc_corr_post = np.nan
    
    # Overall correlations
    if len(df) > 1:
        temp_prod_corr_all = spearmanr(df['Temp_Anomaly'].dropna(), 
                                      df['Prod_Anomaly'].dropna())[0]
        temp_poc_corr_all = spearmanr(df['Temp_Anomaly'].dropna(), 
                                     df['POC_Flux_Anomaly'].dropna())[0]
    else:
        temp_prod_corr_all = np.nan
        temp_poc_corr_all = np.nan
    
    # Create figure for the tables
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # Main anomaly statistics table
    ax1 = fig.add_subplot(gs[0])
    ax1.axis('tight')
    ax1.axis('off')
    
    # Prepare main table data
    table_data = [
        ['Variable', 'Mean Anomaly', '', '', 'Anomaly/Month', '', '', 'Total Anomaly', '', '', 'Units'],
        ['', 'Pre-Blob\n(2011-2013)', 'Blob\n(2014-2016)', 'Post-Blob\n(2017-2021)', 
         'Pre-Blob', 'Blob', 'Post-Blob',
         'Pre-Blob', 'Blob', 'Post-Blob', ''],
    ]
    
    for var, data in stats.items():
        table_data.append([
            var, 
            f"{data['Pre-Blob']:.8f}", 
            f"{data['Blob']:.8f}", 
            f"{data['Post-Blob']:.8f}",
            f"{data['Pre-Blob/Month']:.8f}",
            f"{data['Blob/Month']:.8f}",
            f"{data['Post-Blob/Month']:.8f}",
            f"{data['Pre-Blob Total']:.8f}",
            f"{data['Blob Total']:.8f}",
            f"{data['Post-Blob Total']:.8f}",
            data['Unit']
        ])
    
    # Create main anomaly table
    table1 = ax1.table(cellText=table_data, loc='center', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1, 1.5)
    
    # Format header row
    for j in range(len(table_data[0])):
        table1[(0, j)].set_facecolor('#D7E4F4')
        table1[(1, j)].set_facecolor('#D7E4F4')
    
    # Correlation table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('tight')
    ax2.axis('off')
    
    # Prepare correlation table data
    corr_table_data = [
        ['Spearman Rank Correlation', 'Pre-Blob\n(2011-2013)', 'Blob\n(2014-2016)', 'Post-Blob\n(2017-2021)', 'All Periods\n(2011-2021)'],
        ['Temp Anomaly vs. Productivity Anomaly', 
         f"{temp_prod_corr_pre:.8f}", 
         f"{temp_prod_corr_blob:.8f}", 
         f"{temp_prod_corr_post:.8f}",
         f"{temp_prod_corr_all:.8f}"],
        ['Temp Anomaly vs. POC Flux Anomaly', 
         f"{temp_poc_corr_pre:.8f}", 
         f"{temp_poc_corr_blob:.8f}", 
         f"{temp_poc_corr_post:.8f}",
         f"{temp_poc_corr_all:.8f}"]
    ]
    
    # Create correlation table
    table2 = ax2.table(cellText=corr_table_data, loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1, 1.5)
    
    # Format header row
    for j in range(len(corr_table_data[0])):
        table2[(0, j)].set_facecolor('#D7E4F4')
    
    # Add title
    plt.suptitle(f"Anomaly Statistics: {region_name}", fontsize=16, y=0.98)
    
    # Save figure
    summary_output = os.path.join(output_dir, f"{region_name.replace(' ', '_')}_anomaly_summary.png")
    plt.savefig(summary_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced anomaly summary saved to: {summary_output}")

def save_anomalies_to_csv(df, region_name, output_dir):
    """Save daily anomalies to a CSV file."""
    try:
        if df.empty:
            print(f"No data to save for {region_name}")
            return
            
        print(f"Preparing to save CSV for {region_name}, DataFrame has {len(df)} rows")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output filename with full path
        csv_file = os.path.join(output_dir, f"{region_name.replace(' ', '_')}_daily_anomalies.csv")
        
        # Debug print the full path
        print(f"Attempting to save CSV to: {os.path.abspath(csv_file)}")
        
        # Save to CSV
        df.to_csv(csv_file, index=False, float_format='%.10f')
        print(f"Successfully saved anomalies to: {csv_file}")
        
    except Exception as e:
        print(f"ERROR: Failed to save anomalies for {region_name}: {str(e)}")
        import traceback
        traceback.print_exc()

def calculate_climatology_from_kernels(nc_files, exclude_years=None):
    """
    Calculate daily climatology directly from kernel files.
    
    Parameters:
    -----------
    nc_files : list
        List of netCDF file paths containing monthly data
    exclude_years : list, optional
        List of years to exclude from climatology calculation (e.g., [2014, 2015, 2016] for blob period)
    
    Returns:
    --------
    clim_temp_by_day : dict
        Dictionary mapping day IDs (month*100 + day) to temperature climatology values
    clim_prod_by_day : dict
        Dictionary mapping day IDs to productivity climatology values
    clim_poc_flux_by_day : dict
        Dictionary mapping day IDs to POC flux climatology values
    """
    print("\nCalculating climatology directly from kernel files...")
    
    # Initialize data structures
    day_data = {}  # Format: {day_id: {'temp': [], 'prod': [], 'poc': []}}
    
    # Process each file to extract data by day of year
    for file in nc_files:
        try:
            # Extract month and year from filename
            # Format: z_avg_YYYY_MM_37zlevs_full_1x1meanpool_downsampling_5x5.nc
            # Where MM is month number (001=Jan, 002=Feb, etc.)
            match = re.search(r'z_avg_(\d{4})_(\d{3})_', os.path.basename(file))
            
            if not match:
                print(f"Could not extract date from {file}, skipping...")
                continue
                
            year = int(match.group(1))
            month_str = match.group(2)
            month = int(month_str)

            # Filter out winter months
            if month not in ANALYSIS_MONTHS:
                print(f"Skipping {year}-{month:02d} from climatology: Not in analysis window")
                continue
            
            # Skip files from excluded years
            if exclude_years and year in exclude_years:
                print(f"  Excluding {year}-{month:02d} from climatology (excluded year)")
                continue
            
            print(f"Processing climatology from {year}-{month:02d}: {os.path.basename(file)}")
            
            # Open the dataset
            ds = xr.open_dataset(file)
            ds = process_dataset(ds)
            
            # Check if we have time dimension (daily data within the month)
            if 'time' not in ds.dims:
                print(f"  Warning: No time dimension in {file}, skipping...")
                ds.close()
                continue
                
            # Process each day in the month
            for time_idx in range(ds.dims['time']):
                # Assign day values (1-based) - assuming time index corresponds to day of month
                day = time_idx + 1
                
                # Format day_id as mmdd (e.g., 101 for January 1st)
                day_id = month * 100 + day
                
                # Extract temperature data (top 50m) for this day
                if 'depth' in ds.dims and 'temp' in ds:
                    # Select this day's data and top 50m
                    ds_day = ds.isel(time=time_idx)
                    ds_top = ds_day.where(ds_day['depth'] >= -50, drop=True)
                    
                    if ds_top['depth'].size > 0:
                        # Calculate spatial mean temperature
                        spatial_dims = [dim for dim in ds_top['temp'].dims if dim != 'depth']
                        temp_value = ds_top['temp'].mean(dim=['depth'] + spatial_dims, skipna=True).values.item()
                        
                        # Add to day_data structure
                        if day_id not in day_data:
                            day_data[day_id] = {'temp': [], 'prod': [], 'poc': []}
                        day_data[day_id]['temp'].append(temp_value)
                
                # Extract productivity data (top 50m) for this day
                if 'depth' in ds.dims and 'TOT_PROD' in ds:
                    # Select this day's data and top 50m
                    ds_day = ds.isel(time=time_idx)
                    ds_top = ds_day.where(ds_day['depth'] >= -50, drop=True)
                    
                    if ds_top['depth'].size > 0:
                        # Calculate spatial mean productivity
                        spatial_dims = [dim for dim in ds_top['TOT_PROD'].dims if dim != 'depth']
                        prod_value = ds_top['TOT_PROD'].mean(dim=['depth'] + spatial_dims, skipna=True).values.item()
                        
                        # Add to day_data structure
                        if day_id not in day_data:
                            day_data[day_id] = {'temp': [], 'prod': [], 'poc': []}
                        day_data[day_id]['prod'].append(prod_value)
                
                # Extract POC flux data (at ~100m) for this day
                if 'depth' in ds.dims and 'POC_FLUX_IN' in ds:
                    # Select this day's data
                    ds_day = ds.isel(time=time_idx)
                    
                    # Find closest depth to 100m
                    target_depth = -100
                    depth_idx = np.abs(ds_day['depth'].values - target_depth).argmin()
                    
                    
                    # Find closest depth to 100m
                    target_depth = -100
                    depth_idx = np.abs(ds_day['depth'].values - target_depth).argmin()
                    
                    # Extract POC flux at this depth
                    poc_flux_at_100m = ds_day['POC_FLUX_IN'].isel(depth=depth_idx)
                    
                    # Calculate spatial mean
                    spatial_dims = [dim for dim in poc_flux_at_100m.dims]
                    poc_value = poc_flux_at_100m.mean(dim=spatial_dims, skipna=True).values.item()
                    
                    # Add to day_data structure
                    if day_id not in day_data:
                        day_data[day_id] = {'temp': [], 'prod': [], 'poc': []}
                    day_data[day_id]['poc'].append(poc_value)
            
            ds.close()
            
        except Exception as e:
            print(f"Error processing {file} for climatology: {e}")
            traceback.print_exc()
    
    # Calculate climatology from accumulated data
    clim_temp_by_day = {}
    clim_prod_by_day = {}
    clim_poc_flux_by_day = {}
    
    for day_id, values in day_data.items():
        if values['temp']:
            clim_temp_by_day[day_id] = np.nanmean(values['temp'])
        if values['prod']:
            clim_prod_by_day[day_id] = np.nanmean(values['prod'])
        if values['poc']:
            clim_poc_flux_by_day[day_id] = np.nanmean(values['poc'])
    
    # Report results
    print(f"Calculated climatology for {len(clim_temp_by_day)} days (temperature)")
    print(f"Calculated climatology for {len(clim_prod_by_day)} days (productivity)")
    print(f"Calculated climatology for {len(clim_poc_flux_by_day)} days (POC flux)")
    
    return clim_temp_by_day, clim_prod_by_day, clim_poc_flux_by_day

# Define output directory
base_output_dir = os.path.join(os.path.dirname(__file__), "output2")
os.makedirs(base_output_dir, exist_ok=True)

# Update these paths to where your data actually exists
BASE_DATA_DIR = "/nfs/sea/work/bblaser/regions2"
# Comment out the alternative path since we want to use only the primary path
# ALT_DATA_DIR = "/home/bblaser/data/kernels"  # Not needed

# Main processing loop
# Process each region in the dictionary
create_region_map(regions, base_output_dir)

# In the main processing loop, modify the directory verification
for region_key, region_info in regions.items():
    # Store the region name in a variable at the beginning of the loop
    region_name = region_info['name']
    print(f"\n==== Processing region: {region_name} ====")
    
    # Create region-specific output directory
    output_dir = os.path.join(base_output_dir, region_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the region NAME for finding data files
    data_dir = os.path.join(BASE_DATA_DIR, region_name)
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        print(f"No valid data found for {region_name}, skipping...")
        continue
    
    print(f"Using data directory: {data_dir}")
    # Define file pattern with more flexible matching
    file_pattern = os.path.join(data_dir, "z_avg_*.nc")
    nc_files = sorted(glob.glob(file_pattern))
    
    print(f"Found {len(nc_files)} data files for {region_name}")
    
    if not nc_files:
        print(f"No data files found for {region_name}, skipping...")
        continue
        
    # Calculate climatology directly from kernel files
    # Exclude blob years from climatology calculation
    exclude_years = []  # Blob period years
    original_clim_temp_by_month, original_clim_prod_by_month, original_clim_poc_flux_by_month = calculate_climatology_from_kernels(
        nc_files, 
        exclude_years=exclude_years
    )
    
    # Process all data files for this region using the calculated climatology
    all_data_df = process_data_files(nc_files, original_clim_temp_by_month, original_clim_prod_by_month, original_clim_poc_flux_by_month)
    
    if all_data_df.empty:
        print(f"No valid data found for {region_name}, skipping...")
        continue
    
    # Calculate new climatology by month excluding the Blob period
    print("\nRecalculating climatology excluding Blob period (2014-2016)...")
    new_clim_temp_by_month = {}
    new_clim_prod_by_month = {}
    new_clim_poc_flux_by_month = {}
    
    # Filter out data from 2014-2016
    non_blob_df = all_data_df[~all_data_df['Date'].dt.year.between(2014, 2016)]
    print(f"Using {len(non_blob_df)} months for climatology (excluding 2014-2016)")
    
    # Calculate monthly means from non-blob data
    for month in range(1, 13):
        month_data = non_blob_df[non_blob_df['Date'].dt.month == month]
        if not month_data.empty:
            new_clim_temp_by_month[month] = month_data['Temperature'].mean()
            new_clim_prod_by_month[month] = month_data['TOT_PROD'].mean()
            new_clim_poc_flux_by_month[month] = month_data['POC_FLUX_IN'].mean()
            print(f"Month {month}: {len(month_data)} data points used for climatology")
    
    # Compare old and new climatology
    print("\nComparison of climatologies (with vs without Blob period):")
    for month in range(1, 13):
        old_temp = original_clim_temp_by_month.get(month, np.nan)
        new_temp = new_clim_temp_by_month.get(month, np.nan)
        temp_diff = new_temp - old_temp if not np.isnan(new_temp) and not np.isnan(old_temp) else np.nan
        
        old_prod = original_clim_prod_by_month.get(month, np.nan)
        new_prod = new_clim_prod_by_month.get(month, np.nan)
        prod_diff = new_prod - old_prod if not np.isnan(new_prod) and not np.isnan(old_prod) else np.nan
        
        print(f"Month {month} - Temp: {old_temp:.4f} → {new_temp:.4f} (Δ: {temp_diff:.4f}°C), "
              f"Prod: {old_prod:.10f} → {new_prod:.10f} (Δ: {prod_diff:.10f})")
    
    # Use the new climatology for subsequent analysis
    df = all_data_df.copy()
    
    # Recalculate anomalies based on new climatology
    for i, row in df.iterrows():
        month = row['Date'].month
        
        # Check for Temperature - only calculate if value is not NaN or NaT
        if month in new_clim_temp_by_month and pd.notna(row['Temperature']):
            df.at[i, 'Temp_Anomaly'] = row['Temperature'] - new_clim_temp_by_month[month]
        else:
            df.at[i, 'Temp_Anomaly'] = np.nan
        
        # Check for TOT_PROD
        if month in new_clim_prod_by_month and pd.notna(row['TOT_PROD']):
            df.at[i, 'Prod_Anomaly'] = row['TOT_PROD'] - new_clim_prod_by_month[month]
        else:
            df.at[i, 'Prod_Anomaly'] = np.nan
        
        # Check for POC_FLUX_IN
        if month in new_clim_poc_flux_by_month and pd.notna(row['POC_FLUX_IN']):
            df.at[i, 'POC_Flux_Anomaly'] = row['POC_FLUX_IN'] - new_clim_poc_flux_by_month[month]
        else:
            df.at[i, 'POC_Flux_Anomaly'] = np.nan
    
    # Generate climatology series for plotting
    clim_df = generate_climatology_series(df, new_clim_temp_by_month, new_clim_prod_by_month, new_clim_poc_flux_by_month)
    
    # Print results
    if not df.empty:
        region_display = region_info['name']
        print(f"\nSummary for {region_display}:")
        print(f"Time range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Months of data: {len(df)}")
        print(f"Mean temperature: {df['Temperature'].mean():.4f}°C")
        print(f"Mean productivity: {df['TOT_PROD'].mean() * (3600*24*365.25/1000):.10f} mol/m³/year")
        print(f"Mean POC flux: {df['POC_FLUX_IN'].mean() * (3600*24*365.25/1000):.10f} mol/m²/year")
    
    # Create plots
    region_display = region_info['name']
    
    # Create figure for temperature
    fig_temp = plt.figure(figsize=(30, 12))
    gs_temp = GridSpec(1, 2, figure=fig_temp)
    
    # Create figure for productivity
    fig_prod = plt.figure(figsize=(30, 12))
    gs_prod = GridSpec(1, 2, figure=fig_prod)
    
    # Create figure for POC flux
    fig_poc = plt.figure(figsize=(30, 12))
    gs_poc = GridSpec(1, 2, figure=fig_poc)
    
    # Temperature plots
    ax_temp_compare = fig_temp.add_subplot(gs_temp[0, 0])
    ax_temp_anomaly = fig_temp.add_subplot(gs_temp[0, 1])
    plot_region_data(ax_temp_compare, ax_temp_anomaly, df, clim_df, 
                   region_display, var_type='temp')
    
    # Productivity plots
    ax_prod_compare = fig_prod.add_subplot(gs_prod[0, 0])
    ax_prod_anomaly = fig_prod.add_subplot(gs_prod[0, 1])
    plot_region_data(ax_prod_compare, ax_prod_anomaly, df, clim_df, 
                   region_display, var_type='prod')
    
    # POC flux plots
    ax_poc_compare = fig_poc.add_subplot(gs_poc[0, 0])
    ax_poc_anomaly = fig_poc.add_subplot(gs_poc[0, 1])
    plot_region_data(ax_poc_compare, ax_poc_anomaly, df, clim_df, 
                   region_display, var_type='poc')
    
    # Add titles to figures
    fig_temp.suptitle(f'{region_display} - Ocean Temperature Analysis (Top 50m) - 2011-2021', fontsize=14)
    fig_prod.suptitle(f'{region_display} - Ocean Productivity Analysis (Top 50m) - 2011-2021', fontsize=14)
    fig_poc.suptitle(f'{region_display} - Ocean POC Flux Analysis (100m) - 2011-2021', fontsize=14)
    
    # Adjust layout and save
    fig_temp.tight_layout(rect=[0, 0, 1, 0.95])
    fig_prod.tight_layout(rect=[0, 0, 1, 0.95])
    fig_poc.tight_layout(rect=[0, 0, 1, 0.95])
    
    temp_output = os.path.join(output_dir, f"{region_name}_temperature_analysis.png")
    prod_output = os.path.join(output_dir, f"{region_name}_productivity_analysis.png")
    poc_output = os.path.join(output_dir, f"{region_name}_poc_flux_analysis.png")
    
    fig_temp.savefig(temp_output, dpi=300)
    fig_prod.savefig(prod_output, dpi=300)
    fig_poc.savefig(poc_output, dpi=300)
    
    print(f"\nTemperature plots saved to: {temp_output}")
    print(f"Productivity plots saved to: {prod_output}")
    print(f"POC flux plots saved to: {poc_output}")
    
    # Create a combined figure with all variables together
    fig_combined = plt.figure(figsize=(combined_width, combined_height))
    gs_combined = GridSpec(3, 2, figure=fig_combined, 
                          hspace=subplot_hspace, 
                          wspace=subplot_wspace, 
                          height_ratios=[1, 1, 1])
    
    # Row 1: Temperature plots
    ax_temp_compare_combined = fig_combined.add_subplot(gs_combined[0, 0])
    ax_temp_anomaly_combined = fig_combined.add_subplot(gs_combined[0, 1])
    plot_region_data(ax_temp_compare_combined, ax_temp_anomaly_combined, df, clim_df, 
                    region_display, var_type='temp')
    
    # Row 2: Productivity plots
    ax_prod_compare_combined = fig_combined.add_subplot(gs_combined[1, 0])
    ax_prod_anomaly_combined = fig_combined.add_subplot(gs_combined[1, 1])
    plot_region_data(ax_prod_compare_combined, ax_prod_anomaly_combined, df, clim_df, 
                    region_display, var_type='prod')
    
    # Row 3: POC Flux plots
    ax_poc_compare_combined = fig_combined.add_subplot(gs_combined[2, 0])
    ax_poc_anomaly_combined = fig_combined.add_subplot(gs_combined[2, 1])
    plot_region_data(ax_poc_compare_combined, ax_poc_anomaly_combined, df, clim_df, 
                    region_display, var_type='poc')
    
    # Add overall title with adjusted y position
    fig_combined.suptitle(f'{region_display} - Ocean Analysis (2011-2021)', 
                          fontsize=16, y=suptitle_y)
    
    # Adjust layout and save
    fig_combined.tight_layout(rect=combined_rect)
    combined_output = os.path.join(output_dir, f"{region_name}_combined_analysis.png")
    fig_combined.savefig(combined_output, dpi=300)
    print(f"\nCombined analysis plot saved to: {combined_output}")
    
    plt.close('all')  # Close all figures to free up memory

    # Create anomaly summary
    create_anomaly_summary(df, region_display, output_dir)

# Save anomalies to CSV
    save_anomalies_to_csv(df, region_display, output_dir)

print("\nAll regions processed successfully!")
# %%