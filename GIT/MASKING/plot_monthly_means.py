"""
Calculating statistics of the masks generate by timeseries script.

Author: Beni Blaser  
This script was developed with assistance from Claude Sonnet 3.7 (via Copilot), which supported code creation, debugging, and documentation.
"""
#%%
# Import libraries
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
from matplotlib.table import Table

# Function to determine lat/lon bounds from NetCDF files
def get_lat_lon_bounds(nc_file):
    """
    Extract latitude and longitude bounds from a NetCDF file.
    
    Parameters:
    nc_file (str): Path to NetCDF file
    
    Returns:
    tuple: (lon_min, lon_max, lat_min, lat_max)
    """
    try:
        ds = xr.open_dataset(nc_file)
        
        # Check which coordinate names are present in the dataset
        lon_names = ['lon', 'longitude', 'LONGITUDE', 'lon_rho']
        lat_names = ['lat', 'latitude', 'LATITUDE', 'lat_rho']
        
        lon_var = None
        for name in lon_names:
            if name in ds.coords or name in ds.variables:
                lon_var = name
                break
        
        lat_var = None
        for name in lat_names:
            if name in ds.coords or name in ds.variables:
                lat_var = name
                break
        
        if lon_var is None or lat_var is None:
            print(f"Warning: Could not identify lon/lat coordinates in dataset.")
            print(f"Available variables: {list(ds.variables.keys())}")
            ds.close()
            return None, None, None, None
        
        # Check if coordinates are 1D or 2D
        if len(ds[lon_var].dims) == 2:
            # 2D coordinates (common in ROMS model output)
            lon_min = float(ds[lon_var].min().values)
            lon_max = float(ds[lon_var].max().values)
            lat_min = float(ds[lat_var].min().values)
            lat_max = float(ds[lat_var].max().values)
        else:
            # 1D coordinates
            lon_min = float(ds[lon_var].min().values)
            lon_max = float(ds[lon_var].max().values)
            lat_min = float(ds[lat_var].min().values)
            lat_max = float(ds[lat_var].max().values)
        
        ds.close()
        return lon_min, lon_max, lat_min, lat_max
    
    except Exception as e:
        print(f"Error extracting lat/lon bounds: {e}")
        return None, None, None, None

# Helper function to filter dataset by geographic boundaries
def filter_by_region(ds, lon_min, lon_max, lat_min, lat_max):
    """
    Filter dataset to specified geographic boundaries.
    
    Parameters:
    ds (xarray.Dataset): Input dataset
    
    Returns:
    xarray.Dataset: Filtered dataset
    """
    # Check which coordinate names are present in the dataset
    lon_names = ['lon', 'longitude', 'LONGITUDE', 'lon_rho']
    lat_names = ['lat', 'latitude', 'LATITUDE', 'lat_rho']
    
    lon_var = None
    for name in lon_names:
        if name in ds.coords or name in ds.variables:
            lon_var = name
            break
    
    lat_var = None
    for name in lat_names:
        if name in ds.coords or name in ds.variables:
            lat_var = name
            break
    
    if lon_var is None or lat_var is None:
        print(f"Warning: Could not identify lon/lat coordinates in dataset.")
        print(f"Available variables: {list(ds.variables.keys())}")
        return ds
    
    print(f"  Filtering to region lon={lon_min} to {lon_max}, lat={lat_min} to {lat_max}")
    
    # Check if coordinates are 1D or 2D
    if len(ds[lon_var].dims) == 2:
        # 2D coordinates (common in ROMS model output)
        # Create a mask for points within the region
        lon_mask = (ds[lon_var] >= lon_min) & (ds[lon_var] <= lon_max)
        lat_mask = (ds[lat_var] >= lat_min) & (ds[lat_var] <= lat_max)
        region_mask = lon_mask & lat_mask
        
        # Apply the mask to each variable
        filtered_ds = ds.copy()
        for var_name, var in ds.variables.items():
            # Skip coordinate variables
            if var_name in [lon_var, lat_var]:
                continue
                
            # Check if variable has the spatial dimensions
            spatial_dims = ds[lon_var].dims
            if all(dim in var.dims for dim in spatial_dims):
                # Get the variable's dimensions
                var_dims = var.dims
                
                # Create a boolean mask that matches the shape of the variable
                full_mask = region_mask
                
                # If variable has more dimensions than the mask, we need to broadcast
                # For each dimension in the variable that isn't in the mask
                extra_dims = []
                for dim in var_dims:
                    if dim not in spatial_dims:
                        extra_dims.append(dim)
                
                # Expand the mask for each extra dimension
                for dim in extra_dims:
                    full_mask = full_mask.expand_dims({dim: ds[dim].size})
                
                # Apply the mask to the variable
                filtered_ds[var_name] = var.where(full_mask)
        
        return filtered_ds
    else:
        # 1D coordinates (more typical case)
        return ds.sel({lon_var: slice(lon_min, lon_max), lat_var: slice(lat_min, lat_max)})

# Function to process data files
def process_data_files(nc_files, lon_min, lon_max, lat_min, lat_max):
    """
    Process data files and return a DataFrame with monthly means.
    
    Parameters:
    nc_files (list): List of NetCDF files to process
    
    Returns:
    pd.DataFrame: DataFrame with processed data
    """
    temp_values = []
    tot_prod_values = []
    poc_flux_values = []
    dates = []
    years = []
    months = []
    
    if not nc_files:
        print(f"Warning: No data files found!")
        return pd.DataFrame()
    
    print(f"\nProcessing data files...")
    
    # Sort files chronologically
    nc_files = sorted(nc_files)
    
    for file in nc_files:
        try:
            # Extract date information from filename 
            # Format like z_avg_2018_001_37zlevs_full_1x1meanpool_downsampling.nc
            # Where 001 = January, 002 = February, etc.
            match = re.search(r'z_avg_(\d{4})_(\d{3})_', os.path.basename(file))
            
            if not match:
                print(f"Could not extract year/month from {file}, skipping...")
                continue
                
            year = int(match.group(1))
            month = int(match.group(2))  # This is directly the month number (001=Jan, 002=Feb, etc)
            
            # Create the date directly
            file_date = pd.Timestamp(year=year, month=month, day=1)
            
            print(f"Processing {year}-{month:02d}: {os.path.basename(file)}")
            
            # Open the dataset
            ds = xr.open_dataset(file)
            
            # Filter to study region
            ds = filter_by_region(ds, lon_min, lon_max, lat_min, lat_max)
            
            # Select the top 50 m
            if 'depth' in ds.dims:
                ds_top = ds.where(ds['depth'] >= -50, drop=True)
                if ds_top['depth'].size == 0:
                    print(f"  No depth levels within the top 50 m. Skipping.")
                    ds.close()
                    continue
            else:
                ds_top = ds
            
            # Calculate means for temperature and TOT_PROD
            temp_value = np.nan
            prod_value = np.nan
            poc_flux_value = np.nan
            
            if 'temp' in ds_top:
                # First average over depth (for top 50m only)
                if 'depth' in ds_top['temp'].dims:
                    temp_depth_mean = ds_top['temp'].mean(dim='depth', skipna=True)
                else:
                    temp_depth_mean = ds_top['temp']
                
                # Then average over time if present
                if 'time' in temp_depth_mean.dims:
                    temp_time_mean = temp_depth_mean.mean(dim='time', skipna=True)
                else:
                    temp_time_mean = temp_depth_mean
                
                # Finally average over remaining spatial dimensions
                spatial_dims = [dim for dim in temp_time_mean.dims]
                temp_value = temp_time_mean.mean(dim=spatial_dims, skipna=True).values.item()
            else:
                print(f"  Variable 'temp' not found")
            
            if 'TOT_PROD' in ds_top:
                # First average over depth (for top 50m only)
                if 'depth' in ds_top['TOT_PROD'].dims:
                    prod_depth_mean = ds_top['TOT_PROD'].mean(dim='depth', skipna=True)
                else:
                    prod_depth_mean = ds_top['TOT_PROD']
                
                # Then average over time if present
                if 'time' in prod_depth_mean.dims:
                    prod_time_mean = prod_depth_mean.mean(dim='time', skipna=True)
                else:
                    prod_time_mean = prod_depth_mean
                
                # Finally average over remaining spatial dimensions
                spatial_dims = [dim for dim in prod_time_mean.dims]
                prod_value = prod_time_mean.mean(dim=spatial_dims, skipna=True).values.item()
            else:
                print(f"  Variable 'TOT_PROD' not found")
            
            # Calculate POC flux at 100m
            if 'POC_FLUX_IN' in ds:
                # Find the closest depth to 100m
                if 'depth' in ds.dims:
                    # Get the index of the closest depth to -100m
                    target_depth = -100
                    depth_idx = np.abs(ds['depth'].values - target_depth).argmin()
                    closest_depth = ds['depth'].values[depth_idx]
                    
                    print(f"  Using depth {closest_depth}m for POC_FLUX_IN (target: -100m)")
                    
                    # Extract POC_FLUX_IN at this depth
                    poc_flux_at_100m = ds['POC_FLUX_IN'].isel(depth=depth_idx)
                    
                    # Average over time if present
                    if 'time' in poc_flux_at_100m.dims:
                        poc_flux_time_mean = poc_flux_at_100m.mean(dim='time', skipna=True)
                    else:
                        poc_flux_time_mean = poc_flux_at_100m
                    
                    # Finally average over remaining spatial dimensions
                    spatial_dims = [dim for dim in poc_flux_time_mean.dims]
                    poc_flux_value = poc_flux_time_mean.mean(dim=spatial_dims, skipna=True).values.item()
                else:
                    print(f"  Variable 'POC_FLUX_IN' found but no depth dimension")
            else:
                print(f"  Variable 'POC_FLUX_IN' not found")
            
            # Store values
            dates.append(file_date)
            years.append(year)
            months.append(month)
            temp_values.append(temp_value)
            tot_prod_values.append(prod_value)
            poc_flux_values.append(poc_flux_value)
            
            ds.close()
        except Exception as e:
            print(f"Error processing {file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert lists to arrays and create a DataFrame
    if dates:
        df = pd.DataFrame({
            'Date': dates,
            'Year': years,
            'Month': months,
            'Temperature': temp_values,
            'TOT_PROD': tot_prod_values,
            'POC_FLUX_IN': poc_flux_values
        })
        
        # Sort by date
        df = df.sort_values('Date')
        return df
    else:
        return pd.DataFrame()

# Function to calculate monthly climatology excluding specific years
def calculate_monthly_climatology(df, exclude_years=None):
    """
    Calculate monthly climatology from the data.
    
    Parameters:
    df (pd.DataFrame): DataFrame with the data
    exclude_years (list): Years to exclude from climatology calculation
    
    Returns:
    dict: Dictionary with monthly climatology values
    """
    if df.empty:
        return {}, {}, {}
    
    # Filter out excluded years if specified
    if exclude_years:
        df_clim = df[~df['Year'].isin(exclude_years)].copy()
    else:
        df_clim = df.copy()
    
    print(f"\nCalculating monthly climatology (excluding years {exclude_years})")
    print(f"Using {len(df_clim)} out of {len(df)} total data points")
    
    # Calculate monthly means for each variable
    clim_temp_by_month = {}
    clim_prod_by_month = {}
    clim_poc_flux_by_month = {}
    
    for month in range(1, 13):
        month_data = df_clim[df_clim['Month'] == month]
        if not month_data.empty:
            clim_temp_by_month[month] = month_data['Temperature'].mean()
            clim_prod_by_month[month] = month_data['TOT_PROD'].mean()
            clim_poc_flux_by_month[month] = month_data['POC_FLUX_IN'].mean()
            print(f"Month {month}: {len(month_data)} data points, " +
                  f"Temp: {clim_temp_by_month[month]:.4f}°C, " +
                  f"Prod: {clim_prod_by_month[month]:.6f}, " +
                  f"POC: {clim_poc_flux_by_month[month]:.6f}")
        else:
            print(f"Month {month}: No data available")
    
    return clim_temp_by_month, clim_prod_by_month, clim_poc_flux_by_month

# Function to calculate anomalies and create climatology series
def calculate_anomalies(df, clim_temp_by_month, clim_prod_by_month, clim_poc_flux_by_month):
    """
    Calculate anomalies for each data point and generate full climatology time series.
    
    Parameters:
    df (pd.DataFrame): DataFrame with the data
    clim_*_by_month (dict): Monthly climatology dictionaries
    
    Returns:
    tuple: Updated DataFrame with anomalies and climatology DataFrame
    """
    if df.empty:
        return df, pd.DataFrame()
    
    # Add anomaly columns
    df['Temp_Anomaly'] = np.nan
    df['Prod_Anomaly'] = np.nan
    df['POC_Flux_Anomaly'] = np.nan
    
    # Calculate anomalies
    for i, row in df.iterrows():
        month = row['Month']
        if month in clim_temp_by_month:
            df.at[i, 'Temp_Anomaly'] = row['Temperature'] - clim_temp_by_month[month]
        if month in clim_prod_by_month:
            df.at[i, 'Prod_Anomaly'] = row['TOT_PROD'] - clim_prod_by_month[month]
        if month in clim_poc_flux_by_month:
            df.at[i, 'POC_Flux_Anomaly'] = row['POC_FLUX_IN'] - clim_poc_flux_by_month[month]
    
    # Create climatology DataFrame for plotting
    # Generate dates spanning the range of the data
    date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='MS')
    clim_temp_series = []
    clim_prod_series = []
    clim_poc_flux_series = []
    
    for date in date_range:
        month = date.month
        clim_temp_series.append(clim_temp_by_month.get(month, np.nan))
        clim_prod_series.append(clim_prod_by_month.get(month, np.nan))
        clim_poc_flux_series.append(clim_poc_flux_by_month.get(month, np.nan))
    
    clim_df = pd.DataFrame({
        'Date': date_range,
        'Temp_Climatology': clim_temp_series,
        'Prod_Climatology': clim_prod_series,
        'POC_Flux_Climatology': clim_poc_flux_series
    })
    
    return df, clim_df

# Function to plot data and anomalies with standardized y-axis limits
def plot_region_data(ax_compare, ax_anomaly, df, clim_df, region_name, var_type='temp'):
    """Plot comparison and anomaly for a specific variable and region with standardized y-axis limits."""
    conversion_factor = 3600*24*365.25/1000  # seconds to year and mmol to mol
    blob_start = pd.Timestamp('2014-01-01')
    blob_end = pd.Timestamp('2016-12-31')
    
    if df.empty:
        ax_compare.text(0.5, 0.5, f"No data for {region_name}", 
                       ha='center', transform=ax_compare.transAxes)
        ax_anomaly.text(0.5, 0.5, f"No data for {region_name}", 
                       ha='center', transform=ax_anomaly.transAxes)
        return
    
    # Set up variables based on type
    if var_type == 'temp':
        var_label = 'Temperature'
        clim_var = 'Temp_Climatology'
        anomaly_var = 'Temp_Anomaly'
        color = 'blue'
        dark_color = 'darkblue'
        unit = '°C'
        convert = False  # No conversion needed for temperature
        ylim_compare = (0, 22)  # Standardized y-axis for temperature comparison
        ylim_anomaly = (-2.5, 2.5)  # Standardized y-axis for temperature anomaly
    elif var_type == 'poc':
        var_label = 'POC_FLUX_IN'
        clim_var = 'POC_Flux_Climatology'
        anomaly_var = 'POC_Flux_Anomaly'
        color = 'purple'
        dark_color = 'indigo'
        unit = 'mol/m²/year'
        convert = True  # Convert to mol/m²/year
        ylim_compare = (0, 12)  # Standardized y-axis for POC flux comparison
        ylim_anomaly = (-2.5, 2.5)  # Standardized y-axis for POC flux anomaly
    else:  # productivity
        var_label = 'TOT_PROD'
        clim_var = 'Prod_Climatology'
        anomaly_var = 'Prod_Anomaly'
        color = 'green'
        dark_color = 'darkgreen'
        unit = 'mol/m³/year'
        convert = True  # Convert to mol/m³/year
        ylim_compare = (0, 1.5)  # Standardized y-axis for productivity comparison
        ylim_anomaly = (-0.25, 0.25)  # Standardized y-axis for productivity anomaly
    
    # Plot comparison
    if convert:
        ax_compare.plot(df['Date'], df[var_label] * conversion_factor, 'o-', 
                       color=color, alpha=0.8, markersize=5, linewidth=1.5, label=f'Monthly Mean')
        ax_compare.plot(clim_df['Date'], clim_df[clim_var] * conversion_factor, 'o-', 
                       color=dark_color, alpha=0.6, linewidth=2.0, markersize=5, 
                       label='Monthly Climatology')
    else:
        ax_compare.plot(df['Date'], df[var_label], 'o-', 
                       color=color, alpha=0.8, markersize=5, linewidth=1.5, label=f'Monthly Mean')
        ax_compare.plot(clim_df['Date'], clim_df[clim_var], 'o-', 
                       color=dark_color, alpha=0.6, linewidth=2.0, markersize=5, 
                       label='Monthly Climatology')
    
    # Set standardized y-axis limits for comparison plot
    ax_compare.set_ylim(ylim_compare)
    
    # Highlight the blob period
    ax_compare.axvspan(blob_start, blob_end, color='salmon', alpha=0.2, label='Blob Period (Excluded)')
    ax_compare.set_ylabel(f'{var_label} ({unit})', fontsize=12)
    if var_type == 'poc':
        ax_compare.set_title(f'{var_label}: {region_name} (100m)', fontsize=14)
    else:
        ax_compare.set_title(f'{var_label}: {region_name} (Top 50m)', fontsize=14)
    ax_compare.legend(loc='best', fontsize=10)
    ax_compare.grid(True, alpha=0.3)
    
    # Format x-axis
    years = mdates.YearLocator(2)  # show every 2 years
    months = mdates.MonthLocator()
    years_fmt = mdates.DateFormatter('%Y')
    ax_compare.xaxis.set_major_locator(years)
    ax_compare.xaxis.set_major_formatter(years_fmt)
    ax_compare.xaxis.set_minor_locator(months)
    ax_compare.tick_params(axis='both', which='major', labelsize=10)
    
    # Plot anomaly
    if convert:
        ax_anomaly.plot(df['Date'], df[anomaly_var] * conversion_factor, 'o-', 
                       color=color, alpha=0.8, markersize=5, linewidth=1.5, label=f'Monthly Anomaly')
    else:
        ax_anomaly.plot(df['Date'], df[anomaly_var], 'o-', 
                       color=color, alpha=0.8, markersize=5, linewidth=1.5, label=f'Monthly Anomaly')
    
    # Set standardized y-axis limits for anomaly plot
    ax_anomaly.set_ylim(ylim_anomaly)
    
    ax_anomaly.axhline(y=0.0, color='black', linestyle='--', alpha=0.7, label='Climatology Reference')
    ax_anomaly.axvspan(blob_start, blob_end, color='salmon', alpha=0.2, label='Blob Period (Excluded)')
    ax_anomaly.set_ylabel(f'{var_label} Anomaly ({unit})', fontsize=12)
    if var_type == 'poc':
        ax_anomaly.set_title(f'{var_label} Anomaly: {region_name} (100m)', fontsize=14)
    else:
        ax_anomaly.set_title(f'{var_label} Anomaly: {region_name} (Top 50m)', fontsize=14)
    ax_anomaly.legend(loc='best', fontsize=10)
    ax_anomaly.grid(True, alpha=0.3)
    
    # Format x-axis
    ax_anomaly.xaxis.set_major_locator(years)
    ax_anomaly.xaxis.set_major_formatter(years_fmt)
    ax_anomaly.xaxis.set_minor_locator(months)
    ax_anomaly.tick_params(axis='both', which='major', labelsize=10)

# Function to calculate the integral of anomalies for a specific time period
def calculate_anomaly_integral(df, start_date, end_date, anomaly_var):
    """
    Calculate the integral (sum) of anomalies for a specific time period.
    
    Parameters:
    df (pd.DataFrame): DataFrame with anomaly data
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    anomaly_var (str): Name of the anomaly variable
    
    Returns:
    tuple: (Sum of anomalies, Number of months in period)
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    
    # Filter data for the specified time period
    period_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Calculate the sum of anomalies
    anomaly_sum = period_data[anomaly_var].sum()
    
    return anomaly_sum, len(period_data)

# Function to create an anomaly table figure
def create_anomaly_table(df, region_name, output_dir):
    """
    Create a table figure showing anomalies for different time periods.
    
    Parameters:
    df (pd.DataFrame): DataFrame with anomaly data
    region_name (str): Name of the region for the title
    output_dir (str): Directory to save the output figure
    
    Returns:
    matplotlib.figure.Figure: Figure containing the anomaly table
    """
    conversion_factor = 3600*24*365.25/1000  # seconds to year and mmol to mol
    
    # Define time periods
    periods = [
        ("Pre-Blob (2011-2013)", '2011-01-01', '2013-12-31'),
        ("Blob Period (2014-2016)", '2014-01-01', '2016-12-31'),
        ("Post-Blob (2017-2021)", '2017-01-01', '2021-12-31')
    ]
    
    # Collect data for the table
    table_data = []
    for period_name, start, end in periods:
        # Temperature
        temp_sum, temp_n = calculate_anomaly_integral(df, start, end, 'Temp_Anomaly')
        temp_avg = temp_sum/temp_n if temp_n else 0
        
        # Productivity
        prod_sum, prod_n = calculate_anomaly_integral(df, start, end, 'Prod_Anomaly')
        prod_sum_conv = prod_sum * conversion_factor
        prod_avg = prod_sum_conv/prod_n if prod_n else 0
        
        # POC Flux
        poc_sum, poc_n = calculate_anomaly_integral(df, start, end, 'POC_Flux_Anomaly')
        poc_sum_conv = poc_sum * conversion_factor
        poc_avg = poc_sum_conv/poc_n if poc_n else 0
        
        row = [
            period_name, 
            f"{temp_sum:.2f}°C ({temp_avg:.2f}°C/mo)",
            f"{prod_sum_conv:.4f} ({prod_avg:.4f}/mo)",
            f"{poc_sum_conv:.4f} ({poc_avg:.4f}/mo)"
        ]
        table_data.append(row)
    
    # Create figure for table
    fig_table = plt.figure(figsize=(12, 5))
    ax_table = fig_table.add_subplot(111)
    ax_table.axis('off')
    
    # Column labels
    columns = ['Period', 'Temperature (°C)', 'Productivity (mol/m³/yr)', 'POC Flux (mol/m²/yr)']
    
    # Create the table
    table = ax_table.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Set title
    fig_table.suptitle(f'Anomaly Summary - {region_name}', fontsize=14)
    fig_table.tight_layout()
    
    # Save the figure
    table_output = os.path.join(output_dir, f"{region_name.lower().replace(' ', '_')}_anomaly_table.png")
    fig_table.savefig(table_output, dpi=300, bbox_inches='tight')
    
    print(f"\nAnomaly table saved to: {table_output}")
    
    return fig_table

# Add this new function to create a combined table figure
def create_combined_anomaly_tables(region_data_list, output_dir):
    """
    Create a single figure with anomaly tables for multiple regions.
    
    Parameters:
    region_data_list (list): List of tuples (region_name, df) with region data
    output_dir (str): Directory to save the output figure
    
    Returns:
    matplotlib.figure.Figure: Figure containing all anomaly tables
    """
    conversion_factor = 3600*24*365.25/1000  # seconds to year and mmol to mol
    
    # Create figure for tables
    fig = plt.figure(figsize=(12, 15))  # Taller figure to accommodate 3 tables
    
    # Define time periods
    periods = [
        ("Pre-Blob (2011-2013)", '2011-01-01', '2013-12-31'),
        ("Blob Period (2014-2016)", '2014-01-01', '2016-12-31'),
        ("Post-Blob (2017-2021)", '2017-01-01', '2021-12-31')
    ]
    
    # Column labels
    columns = ['Period', 'Temperature (°C)', 'Productivity (mol/m³/yr)', 'POC Flux (mol/m²/yr)']
    
    # Create subplot for each region
    for i, (region_name, df) in enumerate(region_data_list):
        # Create subplot
        ax = fig.add_subplot(3, 1, i+1)
        ax.axis('off')
        
        # Collect data for this region's table
        table_data = []
        for period_name, start, end in periods:
            # Temperature
            temp_sum, temp_n = calculate_anomaly_integral(df, start, end, 'Temp_Anomaly')
            temp_avg = temp_sum/temp_n if temp_n else 0
            
            # Productivity
            prod_sum, prod_n = calculate_anomaly_integral(df, start, end, 'Prod_Anomaly')
            prod_sum_conv = prod_sum * conversion_factor
            prod_avg = prod_sum_conv/prod_n if prod_n else 0
            
            # POC Flux
            poc_sum, poc_n = calculate_anomaly_integral(df, start, end, 'POC_Flux_Anomaly')
            poc_sum_conv = poc_sum * conversion_factor
            poc_avg = poc_sum_conv/poc_n if poc_n else 0
            
            row = [
                period_name, 
                f"{temp_sum:.2f}°C ({temp_avg:.2f}°C/mo)",
                f"{prod_sum_conv:.4f} ({prod_avg:.4f}/mo)",
                f"{poc_sum_conv:.4f} ({poc_avg:.4f}/mo)"
            ]
            table_data.append(row)
        
        # Create the table for this region
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            colWidths=[0.25, 0.25, 0.25, 0.25]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add region name as subtitle
        ax.set_title(f"{region_name}", fontsize=14, pad=20)
    
    # Set overall title
    fig.suptitle('Marine Heatwave Impact: Regional Anomaly Summary', fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)
    
    # Save the figure
    combined_table_output = os.path.join(output_dir, "combined_anomaly_tables.png")
    fig.savefig(combined_table_output, dpi=300, bbox_inches='tight')
    
    print(f"\nCombined anomaly tables saved to: {combined_table_output}")
    
    return fig

# Function to process a single region
def process_region(data_dir, region_name, output_dir):
    """
    Process data for a specific region.
    
    Parameters:
    data_dir (str): Directory containing the data files
    region_name (str): Name of the region for plot titles
    output_dir (str): Directory to save output files
    """
    print(f"\n\n==== Processing {region_name} data ====")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the file pattern
    file_pattern = os.path.join(data_dir, "allblob_*_z_avg_*_*_37zlevs_full_1x1meanpool_downsampling.nc")
    nc_files = sorted(glob.glob(file_pattern))
    
    print(f"Found {len(nc_files)} data files in {data_dir}")
    
    # Read the lat/lon bounds from the first available file
    lon_min, lon_max, lat_min, lat_max = None, None, None, None
    if nc_files:
        lon_min, lon_max, lat_min, lat_max = get_lat_lon_bounds(nc_files[0])

    # If bounds couldn't be determined, use default values
    if None in (lon_min, lon_max, lat_min, lat_max):
        print("Could not determine bounds from files, using default values")
        lon_min, lon_max = 205, 245  # Default values
        lat_min, lat_max = 20, 38    # Default values

    print(f"Study region: lon={lon_min} to {lon_max}, lat={lat_min} to {lat_max}")
    
    # Process data files
    all_data_df = process_data_files(nc_files, lon_min, lon_max, lat_min, lat_max)
    
    if all_data_df.empty:
        print(f"No data found to process for {region_name}. Skipping.")
        return
    
    # Calculate climatologies (excluding 2014-2016)
    exclude_years = [2014, 2015, 2016]
    clim_temp_by_month, clim_prod_by_month, clim_poc_flux_by_month = calculate_monthly_climatology(
        all_data_df, exclude_years)
    
    # Calculate anomalies
    df, clim_df = calculate_anomalies(
        all_data_df, clim_temp_by_month, clim_prod_by_month, clim_poc_flux_by_month)
    
    # Create anomaly table
    create_anomaly_table(df, region_name, output_dir)
    
    # Print summary statistics
    print(f"\nSummary for {region_name}:")
    print(f"Time range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Months of data: {len(df)}")
    print(f"Mean temperature: {df['Temperature'].mean():.4f}°C")
    print(f"Mean productivity: {df['TOT_PROD'].mean() * (3600*24*365.25/1000):.6f} mol/m³/year")
    print(f"Mean POC flux: {df['POC_FLUX_IN'].mean() * (3600*24*365.25/1000):.6f} mol/m²/year")
    
    # Create plots
    # Create combined figure with all variables together
    fig_combined = plt.figure(figsize=(32, 24))
    gs_combined = GridSpec(3, 2, figure=fig_combined, hspace=0.4, wspace=0.25)
    
    # Row 1: Temperature plots
    ax_temp_compare_combined = fig_combined.add_subplot(gs_combined[0, 0])
    ax_temp_anomaly_combined = fig_combined.add_subplot(gs_combined[0, 1])
    plot_region_data(ax_temp_compare_combined, ax_temp_anomaly_combined, df, clim_df, 
                    region_name, var_type='temp')
    
    # Row 2: Productivity plots
    ax_prod_compare_combined = fig_combined.add_subplot(gs_combined[1, 0])
    ax_prod_anomaly_combined = fig_combined.add_subplot(gs_combined[1, 1])
    plot_region_data(ax_prod_compare_combined, ax_prod_anomaly_combined, df, clim_df, 
                    region_name, var_type='prod')
    
    # Row 3: POC Flux plots
    ax_poc_compare_combined = fig_combined.add_subplot(gs_combined[2, 0])
    ax_poc_anomaly_combined = fig_combined.add_subplot(gs_combined[2, 1])
    plot_region_data(ax_poc_compare_combined, ax_poc_anomaly_combined, df, clim_df, 
                    region_name, var_type='poc')
    
    # Add overall title with improved formatting
    fig_combined.suptitle(f'{region_name} Ocean Analysis\nClimatology excludes Blob Period (2014-2016)', 
                        fontsize=18, y=0.98)
    
    # Adjust layout and save with improved margins
    fig_combined.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    region_code = region_name.split()[0].lower()
    combined_output = os.path.join(output_dir, f"{region_code}_combined_analysis.png")
    fig_combined.savefig(combined_output, dpi=300)
    print(f"\nCombined analysis plot saved to: {combined_output}")
    
    plt.close('all')

# Main execution
def main():
    # Define the regions to process
    regions = [
        {
            "data_dir": "/nfs/sea/work/bblaser/z_avg_blob_studyregions/z_avg_meanpool_northblob", 
            "name": "Northern Region", 
            "output_dir": "/nfs/sea/work/bblaser/region_analysis/north_means"
        },
        {
            "data_dir": "/nfs/sea/work/bblaser/z_avg_blob_studyregions/z_avg_meanpool_centralblob", 
            "name": "Central Region", 
            "output_dir": "/nfs/sea/work/bblaser/region_analysis/central_means"
        },
        {
            "data_dir": "/nfs/sea/work/bblaser/z_avg_blob_studyregions/z_avg_meanpool_southblob", 
            "name": "Southern Region", 
            "output_dir": "/nfs/sea/work/bblaser/region_analysis/south_means"
        }
    ]
    
    # Create a common output directory for combined plots
    common_output_dir = "/nfs/sea/work/bblaser/region_analysis/combined"
    os.makedirs(common_output_dir, exist_ok=True)
    
    # Store data for each region to create combined table
    region_data_list = []
    
    # Loop through each region
    for region in regions:
        # Process region and get data
        region_df = process_region_and_return_data(region["data_dir"], region["name"], region["output_dir"])
        
        # If we have data, add it to our list
        if region_df is not None and not region_df.empty:
            region_data_list.append((region["name"], region_df))
    
    # Create combined anomaly tables if we have data from all regions
    if len(region_data_list) == len(regions):
        create_combined_anomaly_tables(region_data_list, common_output_dir)
    else:
        print("Unable to create combined anomaly tables: missing data from some regions")

# Modified process_region function that returns the dataframe
def process_region_and_return_data(data_dir, region_name, output_dir):
    """
    Process data for a specific region and return the dataframe.
    
    Parameters:
    data_dir (str): Directory containing the data files
    region_name (str): Name of the region for plot titles
    output_dir (str): Directory to save output files
    
    Returns:
    pd.DataFrame: Processed dataframe with anomalies
    """
    print(f"\n\n==== Processing {region_name} data ====")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the file pattern
    file_pattern = os.path.join(data_dir, "allblob_*_z_avg_*_*_37zlevs_full_1x1meanpool_downsampling.nc")
    nc_files = sorted(glob.glob(file_pattern))
    
    print(f"Found {len(nc_files)} data files in {data_dir}")
    
    # Read the lat/lon bounds from the first available file
    lon_min, lon_max, lat_min, lat_max = None, None, None, None
    if nc_files:
        lon_min, lon_max, lat_min, lat_max = get_lat_lon_bounds(nc_files[0])

    # If bounds couldn't be determined, use default values
    if None in (lon_min, lon_max, lat_min, lat_max):
        print("Could not determine bounds from files, using default values")
        lon_min, lon_max = 205, 245  # Default values
        lat_min, lat_max = 20, 38    # Default values

    print(f"Study region: lon={lon_min} to {lon_max}, lat={lat_min} to {lat_max}")
    
    # Process data files
    all_data_df = process_data_files(nc_files, lon_min, lon_max, lat_min, lat_max)
    
    if all_data_df.empty:
        print(f"No data found to process for {region_name}. Skipping.")
        return None
    
    # Calculate climatologies (excluding 2014-2016)
    exclude_years = [2014, 2015, 2016]
    clim_temp_by_month, clim_prod_by_month, clim_poc_flux_by_month = calculate_monthly_climatology(
        all_data_df, exclude_years)
    
    # Calculate anomalies
    df, clim_df = calculate_anomalies(
        all_data_df, clim_temp_by_month, clim_prod_by_month, clim_poc_flux_by_month)
    
    # Still create individual anomaly table if needed
    create_anomaly_table(df, region_name, output_dir)
    
    # Create combined figure with all variables together (keep existing functionality)
    # [existing plotting code]
    
    return df

if __name__ == "__main__":
    main()
# %%
