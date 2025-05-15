#!/usr/bin/env python3
"""
Regional time series analysis of anomalies.
Creates a 4x2 grid of subplots, each containing 3 time series plots (Temperature, Production, and POC Flux).
Highlights marine heatwave periods and uses color schemes based on region type.
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import glob
import cmocean as cmo
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Configuration - Paths for data loading and output
BASE_DIR = '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/kernel_scripts/output'
OUTPUT_DIR = '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/kernel_scripts/timeseries_plots'

# Define regions to process (in the order you want them to appear in the plot)
REGIONS = [
    # 'gulf_alaska',  # Commented out
    'highnpp_north', 
    'lownpp_north',
    'highnpp_central', 
    'lownpp_central', 
    'highnpp_south',
    'lownpp_south'
    # 'low_npp_northerngyre'  # Commented out
]

# Variable names and display settings with updated units and more specific titles
ANOMALY_VARS = {
    'Temp_Anomaly': {'title': 'Temperature Daily Anomalies', 'unit': '°C', 'color': 'red'},
    'Prod_Anomaly': {'title': 'Total Production Daily Anomalies', 'unit': 'mol/m³/y', 'color': 'green'},
    'POC_Flux_Anomaly': {'title': 'POC Flux Daily Anomalies', 'unit': 'molC/m²/y', 'color': 'blue'}
}

# Marine heatwave blob period definition
BLOB_START = datetime(2014, 5, 1)
BLOB_END = datetime(2016, 10, 31)

# Color schemes for regions based on type
REGION_COLORS = {
    'highnpp': {'line': plt.get_cmap('cmo.tempo')(0.6)},
    'lownpp': {'line': plt.get_cmap('cmo.matter')(0.6)},
    'alaska': {'line': plt.get_cmap('cmo.dense')(0.6)}
}

def load_region_data(region_name):
    """
    Load CSV data for a specific region.
    
    Parameters:
    -----------
    region_name : str
        Name of the region directory
    
    Returns:
    --------
    pandas.DataFrame or None
        Loaded dataframe or None if file not found
    """
    region_dir = os.path.join(BASE_DIR, region_name)
    csv_file = os.path.join(region_dir, f"{region_name}_daily_anomalies.csv")
    
    if os.path.exists(csv_file):
        print(f"Loading {csv_file}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Convert Date column to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                # Add year and month columns for easier filtering
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
            
            # Apply unit conversions
            if 'Prod_Anomaly' in df.columns:
                # First convert mmol to mol (×0.001) then s to y (×31557600)
                df['Prod_Anomaly'] = df['Prod_Anomaly'] * 0.001 * 31557600  # mmol/m³/s to mol/m³/y

            if 'POC_Flux_Anomaly' in df.columns:
                df['POC_Flux_Anomaly'] = df['POC_Flux_Anomaly'] * 0.001 * 31557600  # mmolC/m²/s to molC/m²/y

            return df
        except Exception as e:
            print(f"Error loading {csv_file}: {str(e)}")
            return None
    else:
        print(f"File not found: {csv_file}")
        return None

def get_region_color(region_name):
    """
    Determine color scheme for a region based on its name.
    
    Parameters:
    -----------
    region_name : str
        Name of the region
    
    Returns:
    --------
    dict
        Dictionary with color information
    """
    if 'highnpp' in region_name:
        return REGION_COLORS['highnpp']
    elif 'lownpp' in region_name or 'low_npp' in region_name:
        return REGION_COLORS['lownpp']
    elif 'alaska' in region_name or 'gulf' in region_name:
        return REGION_COLORS['alaska']
    else:
        return {'line': 'gray'}

def create_region_timeseries_plots():
    """
    Create a 4x2 grid of subplots with timeseries for all regions.
    Each subplot contains 3 time series (Temperature, Production, POC Flux).
    Filters data to include only May-October period and plots as continuous timeline.
    """
    # Load data for all regions
    all_region_data = {}
    for region in REGIONS:
        df = load_region_data(region)
        if df is not None:
            # Filter to include only May to October data (remove gaps)
            df = df[(df['Month'] >= 5) & (df['Month'] <= 10)]
            all_region_data[region] = df
        else:
            print(f"No data available for {region}")
    
    if not all_region_data:
        print("No data available for plotting")
        return
    
    # Create figure with 3x2 grid (6 regions)
    fig = plt.figure(figsize=(20, 24))  # Adjusted height for 3 rows instead of 4
    # Main grid: 3 rows, 2 columns
    outer_grid = GridSpec(3, 2, figure=fig, hspace=0.2, wspace=0.1)
    
    # Find global min/max for each variable across all regions
    global_ranges = {}
    for var in ANOMALY_VARS.keys():
        min_vals = []
        max_vals = []
        for df in all_region_data.values():
            if var in df.columns:
                # Remove outliers for better scaling (5th to 95th percentile)
                q_low = df[var].quantile(0.05)
                q_high = df[var].quantile(0.95)
                filtered_values = df[var][(df[var] >= q_low) & (df[var] <= q_high)]
                
                if not filtered_values.empty:
                    min_vals.append(filtered_values.min())
                    max_vals.append(filtered_values.max())
        
        if min_vals and max_vals:
            global_ranges[var] = (min(min_vals), max(max_vals))
        else:
            global_ranges[var] = (-1, 1)  # Default range if no data
    
    # OVERRIDE with manual ranges if desired
    manual_ranges = {
        'Temp_Anomaly': (-3, 4),       # Custom range for temperature
        'Prod_Anomaly': (-1, 1),     # Custom range for production
        'POC_Flux_Anomaly': (-8, 8)  # Custom range for POC flux
    }

    # Apply manual overrides to global_ranges
    for var, range_values in manual_ranges.items():
        global_ranges[var] = range_values
    
    # Process each region
    for i, region in enumerate(REGIONS):
        row = i // 2  # Integer division for row
        col = i % 2   # Modulo for column
        
        # Get data for this region
        df = all_region_data.get(region)
        if df is None:
            continue
        
        # Get color scheme for this region
        region_color = get_region_color(region)
        
        # Format region name for display with improved readability
        region_display = region.replace('_', ' ')
        region_display = region_display.replace('lownpp', 'Low NPP')
        region_display = region_display.replace('highnpp', 'High NPP')
        region_display = region_display.title()
        # Fix NPP to be uppercase rather than just Npp
        region_display = region_display.replace('Npp', 'BGC')
        
        # Create subplot for this region
        region_subplot = fig.add_subplot(outer_grid[row, col])
        
        # Remove main axis content, we'll use it only for the title
        region_subplot.axis('off')
        region_subplot.set_title(f"{region_display}", fontsize=16, pad=20)
        
        # Create 3 subplots within this subplot area
        inner_grid = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_grid[row, col], 
                                            hspace=0.2)
        
        # Process data to create seamless time series - remove October to May gaps
        # Sort data by date first
        df = df.sort_values('Date')
        
        # Create continuous index by identifying and removing seasonal gaps
        continuous_dates = []
        date_labels = []
        
        # Extract years from data
        years = sorted(df['Year'].unique())
        
        # Process each year to create continuous dates
        continuous_day_count = 0
        seamless_df = pd.DataFrame()
        
        for year_idx, year in enumerate(years):
            # Get data for this year
            year_data = df[df['Year'] == year].copy()
            
            if year_data.empty:
                continue
                
            # Add continuous day column
            year_data['ContinuousDay'] = range(continuous_day_count, 
                                              continuous_day_count + len(year_data))
            continuous_day_count += len(year_data)
            
            # Add to unified dataframe
            seamless_df = pd.concat([seamless_df, year_data])
            
            # Add tick marks for the BEGINNING of this year, not the middle
            start_point = year_data['ContinuousDay'].min()
            continuous_dates.append(start_point)
            date_labels.append(str(year))
        
        # Process each variable
        for j, var_name in enumerate(ANOMALY_VARS.keys()):
            ax = fig.add_subplot(inner_grid[j])
            
            # Skip if no data for this variable
            if var_name not in df.columns:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                continue
                
            # Plot the time series with continuous days instead of actual dates
            ax.plot(seamless_df['ContinuousDay'], seamless_df[var_name], 
                    color=region_color['line'], 
                    linewidth=0.75, alpha=0.8)
            
            # Highlight marine heatwave period (blob) - need to map to continuous days
            blob_period = seamless_df[
                (seamless_df['Date'] >= BLOB_START) & 
                (seamless_df['Date'] <= BLOB_END)
            ]
            
            if not blob_period.empty:
                blob_start_idx = blob_period['ContinuousDay'].min()
                blob_end_idx = blob_period['ContinuousDay'].max()
                ax.axvspan(blob_start_idx, blob_end_idx, alpha=0.1, color='red', 
                         label='Blob Period')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.5)
            
            # Set variable-specific title and labels
            var_info = ANOMALY_VARS[var_name]
            # Add title to all plots
            ax.set_title(f"{var_info['title']}", fontsize=12)  # Removed "Anomaly" since it's in the title
            
            # Add y-axis label
            ax.set_ylabel(f"{var_info['unit']}", fontsize=10)
            
            # Set consistent y-axis limits based on global range
            if var_name in global_ranges:
                y_min, y_max = global_ranges[var_name]
                # Add a small padding
                y_padding = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
            # Configure x-axis with proper tick marks
            if j < 2:  # Remove x-axis labels for top two plots
                ax.set_xticks(continuous_dates)
                ax.set_xticklabels([])
            else:  # Format x-axis for bottom plot
                ax.set_xticks(continuous_dates)
                ax.set_xticklabels(date_labels)
                ax.set_xlabel("Year", fontsize=10)
                ax.tick_params(axis='x', labelsize=8)
            
            # Grid and tick settings
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelsize=8)
            
            # Add legend to top plot only
            if j == 0:
                ax.legend(loc='upper right', fontsize=8)
    
    # Add overall title and adjust layout
    plt.suptitle("Regional Anomaly Time Series (2011-2021, Seasonal Gaps Removed)", fontsize=20, y=0.92)
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save the figure
    output_file = os.path.join(OUTPUT_DIR, "all_regions_seamless_timeseries.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved seamless time series plot to {output_file}")
    plt.show()
    plt.close(fig)

def create_monthly_anomaly_integrals():
    """
    Calculate the monthly integrals (sums) of daily anomalies for each region
    and variable, and save them to a CSV file.
    """
    print("Calculating monthly anomaly integrals...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # List to store integral data
    all_integral_data = []
    
    # Process each region
    for region in REGIONS:
        # Load data for this region
        df = load_region_data(region)
        
        if df is None or df.empty:
            print(f"No data available for {region}")
            continue
        
        # Format region name for display
        region_display = region.replace('_', ' ')
        region_display = region_display.replace('lownpp', 'Low NPP')
        region_display = region_display.replace('highnpp', 'High NPP')
        region_display = region_display.title()
        # Fix NPP to be uppercase rather than just Npp
        region_display = region_display.replace('Npp', 'BGC')
        
        # Group by year and month
        df['YearMonth'] = df['Date'].dt.to_period('M')
        
        # Calculate monthly integrals (sums) for each variable
        monthly_groups = df.groupby('YearMonth')
        
        for group_name, group_data in monthly_groups:
            year = group_name.year
            month = group_name.month
            
            # Create a row for this month
            month_row = {
                'Region': region_display,
                'Year': year,
                'Month': month,
                'Date': f"{year}-{month:02d}-01",  # First day of month
                'Days': len(group_data)  # Number of days with data in this month
            }
            
            # Calculate sum for each variable
            for var_name in ANOMALY_VARS.keys():
                if var_name in group_data.columns:
                    # Calculate monthly total (integral)
                    monthly_sum = group_data[var_name].sum()
                    # Calculate monthly average 
                    monthly_avg = group_data[var_name].mean()
                    
                    # Store both values
                    month_row[f"{var_name}_Sum"] = monthly_sum
                    month_row[f"{var_name}_Avg"] = monthly_avg
                else:
                    month_row[f"{var_name}_Sum"] = np.nan
                    month_row[f"{var_name}_Avg"] = np.nan
            
            # Add to integral data list
            all_integral_data.append(month_row)
    
    # Convert to dataframe
    integral_df = pd.DataFrame(all_integral_data)
    
    # Sort by region, year, month
    if not integral_df.empty:
        integral_df = integral_df.sort_values(['Region', 'Year', 'Month'])
        
        # Convert Date column to datetime
        integral_df['Date'] = pd.to_datetime(integral_df['Date'])
        
        # Save to CSV
        output_file = os.path.join(OUTPUT_DIR, "monthly_anomaly_integrals.csv")
        integral_df.to_csv(output_file, index=False)
        print(f"Saved monthly anomaly integrals to {output_file}")
        
        return integral_df
    else:
        print("No integral data calculated")
        return None

def main():
    """Main function to execute the workflow."""
    print("=" * 80)
    print("REGIONAL ANOMALY TIME SERIES ANALYSIS")
    print("=" * 80)
    
    # Create the time series plots
    create_region_timeseries_plots()
    
    # Calculate and save monthly anomaly integrals
    create_monthly_anomaly_integrals()
    
    print("\nAll analyses completed successfully!")

if __name__ == "__main__":
    main()