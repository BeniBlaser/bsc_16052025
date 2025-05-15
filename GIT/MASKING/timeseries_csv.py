#!/usr/bin/env python3
"""
Plot time series of anomalies (2011-2021) comparing all thresholds and whole region.
Creates continuous timeline plots with no gaps between years for better visualization.
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

#%%
# Configuration
DATA_DIR = '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/surface_mask_timeseries_apr_nov'
OUTPUT_DIR = DATA_DIR
VARIABLES = ['temp', 'TOT_PROD', 'POC_FLUX_IN']
THRESHOLDS = [10, 30, 50]
TITLES = {
    'temp': 'Temperature', 'TOT_PROD': 'NPP', 'POC_FLUX_IN': 'POC Flux'
}
UNITS = {
    'temp': '°C', 'TOT_PROD': 'mol/m³/y', 'POC_FLUX_IN': 'mol/m²/y'
}
COLORS = {
    'whole_region': 'black', 10: 'blue', 30: 'green', 50: 'red'
}

# Blob period definition
BLOB_START = datetime(2014, 5, 1)
BLOB_END = datetime(2016, 10, 31)

def load_data():
    """Load data files for all variables and thresholds."""
    print("Loading anomalies data...")
    all_data = {}
    
    # Load whole region data
    for variable in VARIABLES:
        whole_region_loaded = False
        
        # Check if dedicated whole region file exists
        whole_region_file = os.path.join(DATA_DIR, f"{variable}_data_with_method_comparison_threshold_0.csv")
        
        if os.path.exists(whole_region_file):
            try:
                df = pd.read_csv(whole_region_file)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Add year, month, day columns
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
                
                # Apply conversion factor for TOT_PROD
                if variable == 'TOT_PROD' and 'anomaly' in df.columns:
                    df['anomaly'] = df['anomaly'] * 31536.0
                
                # Filter to include only May-October (months 5-10)
                df = df[(df['date'].dt.month >= 5) & (df['date'].dt.month <= 10)]
                
                all_data[f"{variable}_whole_region"] = df
                whole_region_loaded = True
            except Exception as e:
                print(f"  - ERROR loading {variable}: {str(e)}")
        
        # If whole region file doesn't exist, try to extract from threshold files
        if not whole_region_loaded:
            for threshold in THRESHOLDS:
                threshold_file = os.path.join(DATA_DIR, f"{variable}_data_with_method_comparison_threshold_{threshold}.csv")
                if os.path.exists(threshold_file):
                    try:
                        df = pd.read_csv(threshold_file)
                        
                        # Check if this file contains whole_region data
                        if 'method' in df.columns and 'whole_region' in df['method'].values:
                            # Convert date if needed
                            if 'date' in df.columns:
                                df['date'] = pd.to_datetime(df['date'])
                            
                            # Apply conversion factor for TOT_PROD
                            if variable == 'TOT_PROD' and 'anomaly' in df.columns:
                                df['anomaly'] = df['anomaly'] * 31536.0
                            
                            # Add year, month, day columns
                            df['year'] = df['date'].dt.year
                            df['month'] = df['date'].dt.month
                            df['day'] = df['date'].dt.day
                            
                            # Filter to include only May-October (months 5-10)
                            df = df[(df['date'].dt.month >= 5) & (df['date'].dt.month <= 10)]
                            
                            # Filter only whole_region rows
                            wr_df = df[df['method'] == 'whole_region'].copy()
                            
                            if not wr_df.empty:
                                all_data[f"{variable}_whole_region"] = wr_df
                                whole_region_loaded = True
                                break
                    except Exception as e:
                        print(f"  - ERROR checking {variable} threshold {threshold}: {str(e)}")
    
    # Load threshold data
    for variable in VARIABLES:
        for threshold in THRESHOLDS:
            threshold_file = os.path.join(DATA_DIR, f"{variable}_data_with_method_comparison_threshold_{threshold}.csv")
            
            if os.path.exists(threshold_file) and f"{variable}_threshold_{threshold}" not in all_data:
                try:
                    df = pd.read_csv(threshold_file)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    
                    # Apply conversion factor for TOT_PROD
                    if variable == 'TOT_PROD' and 'anomaly' in df.columns:
                        df['anomaly'] = df['anomaly'] * 31536.0
                    
                    # Add year, month, day columns
                    df['year'] = df['date'].dt.year
                    df['month'] = df['date'].dt.month
                    df['day'] = df['date'].dt.day
                    
                    # Filter to include only May-October (months 5-10)
                    df = df[(df['date'].dt.month >= 5) & (df['date'].dt.month <= 10)]
                    
                    # Filter for MHW mask method and blob period
                    if 'method' in df.columns and 'period' in df.columns:
                        df = df[(df['method'] == 'mhw_mask') & (df['period'] == 'blob')]
                    
                    # Store in all_data if not empty
                    if not df.empty:
                        all_data[f"{variable}_threshold_{threshold}"] = df
                except Exception as e:
                    print(f"  - ERROR loading {variable} threshold {threshold}: {str(e)}")
    
    return all_data

def create_continuous_timeline(data_dict):
    """Convert data to continuous timeline for plotting."""
    continuous_data = {variable: {'data': {}} for variable in VARIABLES}
    
    for variable in VARIABLES:
        # Process whole region data
        wr_key = f"{variable}_whole_region"
        if wr_key in data_dict:
            wr_df = data_dict[wr_key].copy()
            
            # Filter for whole region method if column exists
            if 'method' in wr_df.columns:
                wr_df = wr_df[wr_df['method'] == 'whole_region']
            
            if not wr_df.empty:
                # Sort by date
                wr_df = wr_df.sort_values('date')
                
                # Create continuous position (year-month-day)
                x_values = []
                y_values = []
                
                for _, row in wr_df.iterrows():
                    year = row['date'].year
                    month = row['date'].month
                    day = row['date'].day
                    
                    # Months 5-10 map to positions 0-5
                    month_pos = month - 5
                    
                    # Final position calculation - center days within month
                    pos = (year - 2011) * 6 + month_pos + day/31.0
                    
                    x_values.append(pos)
                    y_values.append(row['anomaly'])
                
                continuous_data[variable]['data']['whole_region'] = {'x': x_values, 'y': y_values}
        
        # Process threshold data
        for threshold in THRESHOLDS:
            th_key = f"{variable}_threshold_{threshold}"
            if th_key in data_dict:
                th_df = data_dict[th_key].copy()
                
                # Keep blob period data specifically, but don't filter by method/period if it would exclude data
                if 'method' in th_df.columns and 'period' in th_df.columns:
                    mhw_df = th_df[(th_df['method'] == 'mhw_mask') & 
                                  (th_df['period'] == 'blob')]
                    
                    # If filtering results in missing data at the end of the blob period, include more data
                    if mhw_df.empty or mhw_df['date'].max() < BLOB_END:
                        # Just filter by date range instead if needed
                        blob_range_df = th_df[(th_df['date'] >= BLOB_START) & 
                                            (th_df['date'] <= BLOB_END)]
                        if not blob_range_df.empty:
                            mhw_df = blob_range_df
                else:
                    mhw_df = th_df
                
                if not mhw_df.empty:
                    # Sort by date
                    mhw_df = mhw_df.sort_values('date')
                    
                    # Create continuous position
                    x_values = []
                    y_values = []
                    
                    for _, row in mhw_df.iterrows():
                        year = row['date'].year
                        month = row['date'].month
                        day = row['date'].day
                        
                        # Months 5-10 map to positions 0-5
                        month_pos = month - 5
                        
                        # Final position calculation - center days within month
                        pos = (year - 2011) * 6 + month_pos + day/31.0
                        
                        x_values.append(pos)
                        y_values.append(row['anomaly'])
                    
                    continuous_data[variable]['data'][f'threshold_{threshold}'] = {'x': x_values, 'y': y_values}
    
    return continuous_data

def create_plots(continuous_data, is_monthly=False):
    """Create plots with continuous timeline."""
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=True)
    plt_type = "Monthly Mean" if is_monthly else "Daily"
    
    # For each variable (each subplot)
    for i, variable in enumerate(VARIABLES):
        ax = axes[i]
        var_title = TITLES.get(variable, variable)
        var_units = UNITS.get(variable, "")
        
        # Plot whole region data
        if 'whole_region' in continuous_data[variable]['data']:
            data = continuous_data[variable]['data']['whole_region']
            
            # Set line width based on plot type
            line_width = 1.0 if is_monthly else 0.75
            
            # Plot WITHOUT markers
            ax.plot(data['x'], data['y'], 
                  color=COLORS['whole_region'], 
                  linewidth=line_width,
                  label="Whole Region", 
                  zorder=5)
        
        # Plot threshold data
        for threshold in THRESHOLDS:
            key = f'threshold_{threshold}'
            if key in continuous_data[variable]['data']:
                data = continuous_data[variable]['data'][key]
                
                # Plot WITHOUT markers
                ax.plot(data['x'], data['y'],
                      color=COLORS.get(threshold, 'gray'),
                      linewidth=1.0 if is_monthly else 0.75,
                      label=f"Threshold {threshold}%")
        
        # Shade blob period
        blob_start = (2014 - 2011) * 6  # May 2014
        blob_end = (2016 - 2011) * 6 + 6  # October 2016
        ax.axvspan(blob_start, blob_end, alpha=0.1, color='red', label='Blob Period')
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Set title and labels
        ax.set_title(f"{var_title} {plt_type} Anomalies", fontsize=14)
        ax.set_ylabel(f"Anomaly ({var_units})")
        ax.grid(True, alpha=0.3)
        
        # Handle x-axis formatting
        years = range(2011, 2022)  # Include 2011-2021
        
        # First add light vertical lines at month boundaries
        for year in range(2011, 2022):
            for month in range(5, 11):  # May through October (5-10)
                month_boundary = (year - 2011) * 6 + (month - 5)
                ax.axvline(x=month_boundary, color='gray', linestyle='-', alpha=0.2, linewidth=0.8)
        
        # Now add stronger season boundary lines at the END of October
        for year in years:
            season_boundary = (year - 2011) * 6 + 6  # End of October
            ax.axvline(x=season_boundary, color='gray', linestyle='-', alpha=0.6, linewidth=1.5)
        
        # Create tick marks for years (centered in each year's growing season)
        tick_positions = []
        tick_labels = []
        
        for year in range(2011, 2022):
            pos = (year - 2011) * 6 + 3  # Middle of the season (between July-August)
            tick_positions.append(pos)
            tick_labels.append(str(year))
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)
        
        # Add month labels (centered in each month)
        month_tick_positions = []
        month_names = ['M', 'J', 'J', 'A', 'S', 'O']  # First letter of each month
        
        for year in years:
            for m_idx in range(6):  # 6 months (May-Oct)
                pos = (year - 2011) * 6 + m_idx + 0.5  # Center of month
                month_tick_positions.append(pos)
        
        # Set minor ticks at month centers
        ax.set_xticks(month_tick_positions, minor=True)
        ax.tick_params(axis='x', which='minor', length=4, width=1, color='gray')
        
        # Set x-axis limits to ensure we include all of October 2021
        ax.set_xlim(0, (2021 - 2011) * 6 + 6)  # End exactly after October 2021
        
        # Add legend for first plot only
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            # Sort legend: Whole Region, thresholds, blob period
            order = ["Whole Region"] + [f"Threshold {t}%" for t in THRESHOLDS] + ["Blob Period"]
            ordered_handles = []
            ordered_labels = []
            
            for label in order:
                if label in labels:
                    idx = labels.index(label)
                    ordered_handles.append(handles[idx])
                    ordered_labels.append(label)
            
            ax.legend(ordered_handles, ordered_labels, loc='upper right')
        
        # Add depth info
        depth_info = "at 100m" if variable == 'POC_FLUX_IN' else "0m-50m mean"
        ax.text(0.02, 0.95, depth_info, transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Create a twin axis for the top x-axis to show years
        if i == 0:  # Only add to the top subplot
            top_ax = ax.twiny()
            top_ax.set_xlim(ax.get_xlim())
            top_ax.set_xticks(tick_positions)
            top_ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)
            top_ax.tick_params(axis='x', which='both', pad=2)
            
            # Remove top_ax spines except for top
            top_ax.spines['bottom'].set_visible(False)
            top_ax.spines['left'].set_visible(False)
            top_ax.spines['right'].set_visible(False)
        
        # For the bottom x-axis
        ax.set_xticks([])  # Remove the major ticks (years)
    
    # Add month labels for the bottom subplot only
    i = len(VARIABLES) - 1  # Bottom subplot
    for year in years:
        for m_idx, month in enumerate(month_names):
            # Keep month labels in the middle of each month
            pos = (year - 2011) * 6 + m_idx + 0.5  # Center of month
            axes[i].text(pos, -0.02, month, transform=axes[i].get_xaxis_transform(),
                      ha='center', va='top', fontsize=8, color='gray')
    
    # Add overall title and final adjustments
    axes[-1].set_xlabel("Months")
    plt_type_txt = "Monthly Mean" if is_monthly else "Daily"
    plt.suptitle(f"Northeast Pacific {plt_type_txt} Anomalies (2011-2021)\nMarine Heatwave Thresholds & Whole Region Comparison", 
               fontsize=16, y=0.99)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    output_file = os.path.join(OUTPUT_DIR, f"{'monthly' if is_monthly else 'daily'}_anomalies_continuous_timeline.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    
    return fig

def calculate_monthly_averages(data_dict):
    """Calculate monthly averages for creating monthly plots."""
    monthly_data = {variable: {'data': {}} for variable in VARIABLES}
    
    print("\nChecking data availability for October 2016:")
    
    for variable in VARIABLES:
        # Process threshold data - check this first for the blob period
        for threshold in THRESHOLDS:
            th_key = f"{variable}_threshold_{threshold}"
            if th_key in data_dict:
                df = data_dict[th_key].copy()
                
                # Check for October 2016 data
                oct_2016_data = df[(df['date'].dt.year == 2016) & (df['date'].dt.month == 10)]
                print(f"  - {variable} threshold {threshold}%: {len(oct_2016_data)} data points for October 2016")
                
                # Get blob period data with extra checks
                if 'method' in df.columns and 'period' in df.columns:
                    mhw_df = df[(df['method'] == 'mhw_mask') & (df['period'] == 'blob')].copy()
                    
                    # Check if we have October 2016 data
                    oct_2016_in_mhw = mhw_df[(mhw_df['date'].dt.year == 2016) & (mhw_df['date'].dt.month == 10)]
                    if len(oct_2016_in_mhw) == 0:
                        # Try without the method/period filter
                        blob_date_df = df[(df['date'] >= BLOB_START) & (df['date'] <= BLOB_END)].copy()
                        oct_2016_in_date = blob_date_df[(blob_date_df['date'].dt.year == 2016) & 
                                                      (blob_date_df['date'].dt.month == 10)]
                        mhw_df = blob_date_df
                else:
                    mhw_df = df[(df['date'] >= BLOB_START) & (df['date'] <= BLOB_END)].copy()
                
                if not mhw_df.empty:
                    # Create year-month groups
                    mhw_df['year_month'] = mhw_df['date'].dt.to_period('M')
                    
                    # Get monthly means
                    monthly_means = mhw_df.groupby('year_month').agg({'anomaly': 'mean', 'date': lambda x: x.iloc[0]})
                    
                    # Check if we have the October 2016 period
                    period_2016_10 = pd.Period('2016-10')
                    if period_2016_10 in monthly_means.index:
                        print(f"    - October 2016 mean exists for {variable} threshold {threshold}%!")
                    
                    # Convert to continuous timeline
                    x_values = []
                    y_values = []
                    
                    for period, row in monthly_means.iterrows():
                        date = row['date']
                        year = date.year
                        month = date.month
                        
                        # Skip months outside May-October
                        if month < 5 or month > 10:
                            continue
                            
                        # Months 5-10 map to positions 0-5
                        month_pos = month - 5
                        
                        # Final position calculation - center in month
                        pos = (year - 2011) * 6 + month_pos + 0.5
                        
                        x_values.append(pos)
                        y_values.append(row['anomaly'])
                    
                    monthly_data[variable]['data'][f'threshold_{threshold}'] = {'x': x_values, 'y': y_values}
        
        # Process whole region data
        wr_key = f"{variable}_whole_region"
        if wr_key in data_dict:
            df = data_dict[wr_key].copy()
            
            # Filter for whole region method if column exists
            if 'method' in df.columns:
                df = df[df['method'] == 'whole_region']
            
            if not df.empty:
                # Create year-month groups and calculate means
                df['year_month'] = df['date'].dt.to_period('M')
                monthly_means = df.groupby('year_month').agg({'anomaly': 'mean', 'date': lambda x: x.iloc[0]})
                
                # Convert to continuous timeline
                x_values = []
                y_values = []
                
                for _, row in monthly_means.iterrows():
                    date = row['date']
                    year = date.year
                    month = date.month
                    
                    # Skip months outside May-October
                    if month < 5 or month > 10:
                        continue
                        
                    # Months 5-10 map to positions 0-5
                    month_pos = month - 5
                    
                    # Final position calculation - center in month
                    pos = (year - 2011) * 6 + month_pos + 0.5
                    
                    x_values.append(pos)
                    y_values.append(row['anomaly'])
                
                monthly_data[variable]['data']['whole_region'] = {'x': x_values, 'y': y_values}
    
    return monthly_data

def calculate_blob_period_statistics(data_dict):
    """
    Calculate and display anomaly means for:
    1. Pre-blob period (whole region only)
    2. Blob period (whole region and all thresholds)
    3. Post-blob period (whole region only)
    """
    results = []
    headers = ["Variable", "Period", "Region/Threshold", "Monthly Anomaly Mean"]
    
    for variable in VARIABLES:
        var_title = TITLES.get(variable, variable)
        
        # Process whole region data
        wr_key = f"{variable}_whole_region"
        if wr_key in data_dict and 'anomaly' in data_dict[wr_key].columns:
            df = data_dict[wr_key].copy()
            
            # Filter for whole region method if column exists
            if 'method' in df.columns:
                df = df[df['method'] == 'whole_region']
            
            # Pre-blob period (2011-2013)
            pre_blob_df = df[df['date'] < BLOB_START].copy()
            if not pre_blob_df.empty:
                pre_blob_df['year_month'] = pre_blob_df['date'].dt.to_period('M')
                monthly_means = pre_blob_df.groupby('year_month')['anomaly'].mean()
                pre_blob_mean = monthly_means.mean()
                
                results.append([
                    var_title,
                    "Pre-Blob",
                    "Whole Region",
                    f"{pre_blob_mean:.4f}" if not np.isnan(pre_blob_mean) else "-"
                ])
            
            # Blob period (2014-2016) for whole region
            blob_df = df[(df['date'] >= BLOB_START) & (df['date'] <= BLOB_END)].copy()
            if not blob_df.empty:
                blob_df['year_month'] = blob_df['date'].dt.to_period('M')
                monthly_means = blob_df.groupby('year_month')['anomaly'].mean()
                blob_mean = monthly_means.mean()
                
                results.append([
                    var_title,
                    "Blob",
                    "Whole Region",
                    f"{blob_mean:.4f}" if not np.isnan(blob_mean) else "-"
                ])
            
            # Post-blob period (2017-2021)
            post_blob_df = df[df['date'] > BLOB_END].copy()
            if not post_blob_df.empty:
                post_blob_df['year_month'] = post_blob_df['date'].dt.to_period('M')
                monthly_means = post_blob_df.groupby('year_month')['anomaly'].mean()
                post_blob_mean = monthly_means.mean()
                
                results.append([
                    var_title,
                    "Post-Blob",
                    "Whole Region",
                    f"{post_blob_mean:.4f}" if not np.isnan(post_blob_mean) else "-"
                ])
        
        # Process threshold data during blob period
        for threshold in THRESHOLDS:
            th_key = f"{variable}_threshold_{threshold}"
            if th_key in data_dict and 'anomaly' in data_dict[th_key].columns:
                df = data_dict[th_key].copy()
                
                # Filter blob period data
                blob_df = df[(df['date'] >= BLOB_START) & (df['date'] <= BLOB_END)].copy()
                if not blob_df.empty:
                    blob_df['year_month'] = blob_df['date'].dt.to_period('M')
                    monthly_means = blob_df.groupby('year_month')['anomaly'].mean()
                    blob_mean = monthly_means.mean()
                    
                    results.append([
                        var_title,
                        "Blob",
                        f"Threshold {threshold}%",
                        f"{blob_mean:.4f}" if not np.isnan(blob_mean) else "-"
                    ])
    
    # Print results as a formatted table
    print("\n" + "="*80)
    print("MONTHLY ANOMALY MEANS BY PERIOD")
    print("="*80)
    
    # Format and print the table
    col_widths = [15, 10, 20, 20]
    
    # Print header
    header_row = "".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    print(header_row)
    print("-" * sum(col_widths))
    
    # Group results by variable for better readability
    for variable in VARIABLES:
        var_title = TITLES.get(variable, variable)
        var_rows = [row for row in results if row[0] == var_title]
        
        for row in var_rows:
            formatted_row = "".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row)))
            print(formatted_row)
        
        # Add separator between variables
        if variable != VARIABLES[-1]:
            print("-" * sum(col_widths))
    
    # Save to CSV
    output_file = os.path.join(OUTPUT_DIR, "blob_period_monthly_statistics.csv")
    with open(output_file, 'w') as f:
        f.write(",".join(headers) + "\n")
        for row in results:
            f.write(",".join(row) + "\n")
    
    print(f"\nStatistics saved to {output_file}")
    
    return results

def calculate_monthly_integrals(data_dict):
    """
    Calculate monthly integrals (sums) of anomalies and save to CSV.
    This represents the accumulated anomaly effect for each month.
    """
    print("\nCalculating monthly anomaly integrals...")
    
    # Prepare data structure for results
    integral_data = []
    
    for variable in VARIABLES:
        var_title = TITLES.get(variable, variable)
        var_unit = UNITS.get(variable, "")
        
        # Process whole region data first
        wr_key = f"{variable}_whole_region"
        if wr_key in data_dict:
            df = data_dict[wr_key].copy()
            
            # Filter for whole region method if column exists
            if 'method' in df.columns:
                df = df[df['method'] == 'whole_region']
            
            if not df.empty:
                # Group by year and month
                df['year_month'] = df['date'].dt.to_period('M')
                
                # Calculate monthly sums and number of data points
                monthly_sums = df.groupby('year_month').agg({
                    'anomaly': ['sum', 'mean', 'count'], 
                    'date': lambda x: x.iloc[0]
                })
                
                # Add to integral data
                for period, row in monthly_sums.iterrows():
                    year = period.year
                    month = period.month
                    
                    # Skip months outside May-October
                    if month < 5 or month > 10:
                        continue
                    
                    # Add to results
                    integral_data.append({
                        'Variable': var_title,
                        'Year': year,
                        'Month': month,
                        'Region': 'Whole Region',
                        'Threshold': 'N/A',
                        'Sum': row['anomaly']['sum'],
                        'Mean': row['anomaly']['mean'],
                        'Days': row['anomaly']['count'],
                        'Unit': var_unit
                    })
        
        # Process threshold data
        for threshold in THRESHOLDS:
            th_key = f"{variable}_threshold_{threshold}"
            if th_key in data_dict:
                df = data_dict[th_key].copy()
                
                # Get blob period data (with appropriate filtering)
                if 'method' in df.columns and 'period' in df.columns:
                    blob_df = df[(df['method'] == 'mhw_mask') & (df['period'] == 'blob')]
                    
                    # If filtering results in missing data, try date-based filtering
                    if blob_df.empty or blob_df['date'].max() < BLOB_END:
                        blob_df = df[(df['date'] >= BLOB_START) & (df['date'] <= BLOB_END)]
                else:
                    blob_df = df[(df['date'] >= BLOB_START) & (df['date'] <= BLOB_END)]
                
                if not blob_df.empty:
                    # Group by year and month
                    blob_df['year_month'] = blob_df['date'].dt.to_period('M')
                    
                    # Calculate monthly sums and number of data points
                    monthly_sums = blob_df.groupby('year_month').agg({
                        'anomaly': ['sum', 'mean', 'count'], 
                        'date': lambda x: x.iloc[0]
                    })
                    
                    # Add to integral data
                    for period, row in monthly_sums.iterrows():
                        year = period.year
                        month = period.month
                        
                        # Skip months outside May-October
                        if month < 5 or month > 10:
                            continue
                        
                        # Add to results
                        integral_data.append({
                            'Variable': var_title,
                            'Year': year,
                            'Month': month,
                            'Region': 'MHW Mask',
                            'Threshold': f"{threshold}%",
                            'Sum': row['anomaly']['sum'],
                            'Mean': row['anomaly']['mean'],
                            'Days': row['anomaly']['count'],
                            'Unit': var_unit
                        })
    
    # Convert to DataFrame and save
    if integral_data:
        integral_df = pd.DataFrame(integral_data)
        
        # Sort by variable, region, year, month
        integral_df = integral_df.sort_values(['Variable', 'Region', 'Threshold', 'Year', 'Month'])
        
        # Save to CSV
        output_file = os.path.join(OUTPUT_DIR, "monthly_anomaly_integrals.csv")
        integral_df.to_csv(output_file, index=False)
        print(f"Monthly anomaly integrals saved to: {output_file}")
        
        # Print summary statistics for blob period
        print("\nBlob Period (2014-2016) Monthly Integral Summaries:")
        blob_period_df = integral_df[(integral_df['Year'] >= 2014) & (integral_df['Year'] <= 2016)]
        
        # Group by variable, region, threshold and calculate total sum
        summaries = blob_period_df.groupby(['Variable', 'Region', 'Threshold']).agg({
            'Sum': 'sum', 
            'Mean': 'mean',
            'Days': 'sum'
        }).reset_index()
        
        # Print top 5 rows of summary
        print(summaries.head().to_string())
        
        return integral_df
    else:
        print("No data available for monthly integral calculation")
        return None

def main():
    """Main function to execute the workflow."""
    print("=" * 80)
    print("ANOMALIES TIME SERIES (2011-2021): CONTINUOUS TIMELINE VISUALIZATION")
    print("=" * 80)
    
    # Step 1: Load all data
    all_data = load_data()
    
    if not all_data:
        print("Error: No data available. Check data directory and file patterns.")
        return
    
    # Step 2: Create daily plot with continuous timeline
    print("\nCreating daily anomaly plots with continuous timeline...")
    continuous_data = create_continuous_timeline(all_data)
    create_plots(continuous_data, is_monthly=False)
    
    # Step 3: Calculate monthly averages and create monthly plot
    print("\nCreating monthly average anomaly plots with continuous timeline...")
    monthly_data = calculate_monthly_averages(all_data)
    create_plots(monthly_data, is_monthly=True)
    
    # Step 4: Calculate and display blob period statistics
    print("\nCalculating pre-blob, blob period, and post-blob statistics...")
    calculate_blob_period_statistics(all_data)
    
    # Step 5: Calculate and save monthly integrals
    print("\nCalculating and saving monthly integrals...")
    calculate_monthly_integrals(all_data)
    
    print("\nAll plots and tables created successfully!")

if __name__ == "__main__":
    main()