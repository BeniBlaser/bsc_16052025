#!/usr/bin/env python3
"""
Regional correlation analysis script for anomaly data with explicit CSV paths.
Creates a combined 8x3 plot showing all regions and their correlation patterns 
for Temperature, Production, and POC Flux anomalies.
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import glob
import cmocean as cmo
from scipy import stats
import matplotlib
from statsmodels.nonparametric.smoothers_lowess import lowess

# Configuration - Base paths
BASE_DIR = '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/kernel_scripts/output2'
OUTPUT_DIR = '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/kernel_scripts/timeseries_plots'

# Define regions and their EXACT CSV file paths
REGION_CSV_PATHS = {
    #'gulf_alaska': '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/kernel_scripts/output2/gulf_alaska/gulf_alaska_daily_anomalies.csv',
    #'low_npp_northerngyre': '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/kernel_scripts/output2/low_npp_northerngyre/low_npp_northerngyre_daily_anomalies.csv',    
    'highnpp_north': '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/kernel_scripts/output2/highnpp_north/highnpp_north_daily_anomalies.csv',
    'lownpp_north': '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/kernel_scripts/output2/lownpp_north/lownpp_north_daily_anomalies.csv',
    'highnpp_central': '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/kernel_scripts/output2/highnpp_central/highnpp_central_daily_anomalies.csv',
    'lownpp_central': '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/kernel_scripts/output2/lownpp_central/lownpp_central_daily_anomalies.csv',
    'highnpp_south': '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/kernel_scripts/output2/highnpp_south/highnpp_south_daily_anomalies.csv',
    'lownpp_south': '/home/bblaser/scripts/Beni_Scripts/analyze_extreme_events/kernel_scripts/output2/lownpp_south/lownpp_south_daily_anomalies.csv',
}

# Define regions to process in display order
REGIONS = list(REGION_CSV_PATHS.keys())

# Variable names and display settings
ANOMALY_VARS = {
    'Temp_Anomaly': {'title': 'Temperature Anomaly', 'unit': '°C'},
    'Prod_Anomaly': {'title': 'NPP Anomaly', 'unit': 'mol/m³/y'},  # Changed from /s to /y
    'POC_Flux_Anomaly': {'title': 'POC Flux Anomaly', 'unit': 'mol/m²/y'}  # Changed from /s to /y
}

# Pairs to analyze
CORRELATION_PAIRS = [
    ('Temp_Anomaly', 'Prod_Anomaly'),
    ('Temp_Anomaly', 'POC_Flux_Anomaly'),
    ('Prod_Anomaly', 'POC_Flux_Anomaly')
]

# Color maps based on region type
REGION_COLORMAPS = {
    'highnpp': 'cmo.tempo',  # Green for high npp regions
    'lownpp': 'cmo.matter',  # Red for low npp regions
    'alaska': 'cmo.dense'    # Blue for gulf of alaska
}

# Marine heatwave blob period definition
BLOB_START = datetime(2014, 5, 1)
BLOB_END = datetime(2016, 10, 31)

def load_region_data(region_name):
    """
    Load anomaly data for a specific region using explicit CSV path.
    
    Parameters:
    -----------
    region_name : str
        Name of the region
    
    Returns:
    --------
    pandas.DataFrame or None
        Loaded dataframe or None if file not found
    """
    # Get the explicit CSV path for this region
    csv_file = REGION_CSV_PATHS.get(region_name)
    
    if not csv_file:
        print(f"No CSV path defined for region: {region_name}")
        return None
    
    print(f"Loading data for {region_name} from: {csv_file}")
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            
            # Convert Date column to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                # Add year and month columns for easier filtering
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
            
            # Apply unit conversions
            # Convert production from mmol/m³/s to mol/m³/y
            if 'Prod_Anomaly' in df.columns:
                # First convert mmol to mol (×0.001) then s to y (×31557600)
                df['Prod_Anomaly'] = df['Prod_Anomaly'] * 0.001 * 31557600  # mmol/m³/s to mol/m³/y
            
            # Convert POC flux from mmolC/m²/s to molC/m²/y
            if 'POC_Flux_Anomaly' in df.columns:
                df['POC_Flux_Anomaly'] = df['POC_Flux_Anomaly'] * 0.001 * 31557600  # mmolC/m²/s to molC/m²/y
            
            # Filter for marine heatwave period (2014-2016)
            blob_period = df[(df['Date'] >= BLOB_START) & (df['Date'] <= BLOB_END)]
            
            # Check if we have data in the blob period
            if blob_period.empty:
                print(f"  Warning: No data in the marine heatwave period for {region_name}")
                return df  # Return full data if no blob period data
            
            print(f"  Successfully loaded {len(blob_period)} records for marine heatwave period")
            return blob_period
        except Exception as e:
            print(f"Error loading {csv_file}: {str(e)}")
            return None
    else:
        print(f"File not found: {csv_file}")
        return None

def calculate_correlation(df, var1, var2):
    """
    Calculate correlation statistics between two variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing the variables
    var1, var2 : str
        Names of variables to correlate
    
    Returns:
    --------
    dict
        Dictionary containing correlation statistics
    """
    # Drop rows with NaN values in either variable
    data = df[[var1, var2]].dropna()
    
    if len(data) < 3:
        return {
            'spearman': None,
            'p_value': None,
            'ci_lower': None,
            'ci_upper': None,
            'n': 0,
            'linear_reg': {'slope': None, 'intercept': None, 'r_value': None}
        }
    
    # Calculate Spearman correlation with confidence intervals
    spearman_results = calculate_spearman_ci_fisher(
        np.array(data[var1]), 
        np.array(data[var2])
    )
    
    # Calculate linear regression
    x = np.array(data[var1])
    y = np.array(data[var2])
    
    try:
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
    except Exception:
        slope, intercept, r_value = None, None, None
    
    return {
        'spearman': spearman_results['spearman'],
        'p_value': spearman_results['p_value'],
        'ci_lower': spearman_results['ci_lower'],
        'ci_upper': spearman_results['ci_upper'],
        'n': spearman_results['n'],
        'linear_reg': {'slope': slope, 'intercept': intercept, 'r_value': r_value}
    }

def get_region_colormap(region_name):
    """
    Determine which colormap to use based on region name.
    
    Parameters:
    -----------
    region_name : str
        Name of the region
    
    Returns:
    --------
    str
        Name of the colormap to use
    """
    if 'highnpp' in region_name:
        return REGION_COLORMAPS['highnpp']
    elif 'lownpp' in region_name or 'low_npp' in region_name:
        return REGION_COLORMAPS['lownpp']
    elif 'alaska' in region_name:
        return REGION_COLORMAPS['alaska']
    else:
        return 'viridis'  # Default colormap

def create_combined_correlation_plot(all_region_data):
    """
    Create a combined 6x3 correlation plot for all regions with improved readability.
    No yearly color differentiation.
    """
    if not all_region_data:
        print("No region data available for plotting")
        return
    
    # Set up the figure - 6 rows (regions) by 3 columns (correlation pairs)
    fig, axes = plt.subplots(6, 3, figsize=(16, 20))  # Adjusted figure height for 6 rows

    # Add only one title over the middle plot in the first row
    fig.suptitle("Anomaly Relationships in Northeast Pacific Regions During Marine Heatwave (2014-2016)", 
                 fontsize=22, y=0.98)

    # Set global font sizes
    plt.rcParams.update({
        'font.size': 12,              # Base font size
        'axes.titlesize': 14,         # Subplot titles
        'axes.labelsize': 12,         # Axis labels
        'xtick.labelsize': 11,        # X tick labels
        'ytick.labelsize': 11,        # Y tick labels
        'legend.fontsize': 11,        # Legend text
        'font.family': 'sans-serif',
    })

    # Add title only to the middle plot in the first row
    axes[0, 1].set_title("Northeast Pacific Temperature, NPP, and POC Flux Correlations\nDuring Marine Heatwave (2014-2016)", 
                       fontsize=16, pad=20)
    
    # Dictionary to store min/max values for each variable to ensure consistent axes
    data_ranges = {}
    for var in ANOMALY_VARS:
        all_values = []
        for df in all_region_data.values():
            if df is not None and not df.empty and var in df.columns:
                all_values.extend(df[var].dropna().tolist())
        
        if all_values:
            data_ranges[var] = (min(all_values), max(all_values))
        else:
            data_ranges[var] = (-1, 1)  # Default range if no data
    
    # Process each region
    for row, region in enumerate(REGIONS):
        # Format region name for display with improved readability
        region_display = region.replace('_', ' ')
        region_display = region_display.replace('lownpp', 'Low NPP')
        region_display = region_display.replace('highnpp', 'High NPP')
        region_display = region_display.title()
        region_display = region_display.replace('Npp', 'BGC')
        
        # Get data for this region
        df = all_region_data.get(region)
        
        # Get colormap for this region
        cmap_name = get_region_colormap(region)
        
        # Get a single color for this region type (using middle value 0.6)
        region_color = plt.get_cmap(cmap_name)(0.6)
        
        # Process each correlation pair (column)
        for col, (var1, var2) in enumerate(CORRELATION_PAIRS):
            ax = axes[row, col]
            
            # Add region title with improved formatting
            ax.set_title(f"{region_display}", fontsize=13)
            
            # Skip if no data for this region
            if df is None or df.empty:
                ax.text(0.5, 0.5, "No data available", 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12)
                continue
            
            # Calculate correlation with CI
            corr_stats = calculate_correlation(df, var1, var2)
            
            # Extract data for plotting
            data = df[[var1, var2]].dropna()
            
            if data.empty:
                ax.text(0.5, 0.5, "No data available", 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12)
                continue
            
            # Plot scatter points with a single color for the region
            ax.scatter(data[var1], data[var2], 
                      color=region_color, edgecolor='black', 
                      alpha=0.7, s=30)
            
            # Calculate percentage of points in each quadrant
            x_vals = np.array(data[var1])
            y_vals = np.array(data[var2])
            total_points = len(x_vals)

            # Define the four quadrants
            q1 = np.sum((x_vals > 0) & (y_vals > 0))  # top right
            q2 = np.sum((x_vals < 0) & (y_vals > 0))  # top left
            q3 = np.sum((x_vals < 0) & (y_vals < 0))  # bottom left
            q4 = np.sum((x_vals > 0) & (y_vals < 0))  # bottom right

            # Calculate percentages
            q1_pct = 100 * q1 / total_points if total_points > 0 else 0
            q2_pct = 100 * q2 / total_points if total_points > 0 else 0
            q3_pct = 100 * q3 / total_points if total_points > 0 else 0
            q4_pct = 100 * q4 / total_points if total_points > 0 else 0

            # Position for quadrant labels (adjust as needed based on your data ranges)
            x_min, x_max = data_ranges[var1]
            y_min, y_max = data_ranges[var2]
            x_range = x_max - x_min
            y_range = y_max - y_min

            # Add percentage labels in each quadrant
            if q1 > 0:  # Top right
                ax.text(x_min + 0.75*x_range, y_min + 0.75*y_range, 
                        f"{q1_pct:.1f}%", ha='center', va='center', 
                        fontsize=10, color='black',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.2", edgecolor='none'))

            if q2 > 0:  # Top left
                ax.text(x_min + 0.25*x_range, y_min + 0.75*y_range, 
                        f"{q2_pct:.0f}%", ha='center', va='center', 
                        fontsize=10, color='black',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.2", edgecolor='none'))

            if q3 > 0:  # Bottom left
                ax.text(x_min + 0.25*x_range, y_min + 0.25*y_range, 
                        f"{q3_pct:.0f}%", ha='center', va='center', 
                        fontsize=10, color='black',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.2", edgecolor='none'))

            if q4 > 0:  # Bottom right
                ax.text(x_min + 0.75*x_range, y_min + 0.25*y_range, 
                        f"{q4_pct:.0f}%", ha='center', va='center', 
                        fontsize=10, color='black',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.2", edgecolor='none'))
            
            # Format correlation statistics text, remove CI and sample size
            if corr_stats['spearman'] is not None:
                # Only show Spearman correlation (no CI)
                spearman_text = f"ρ = {corr_stats['spearman']:.3f}"
                
                # Add p-value with significance stars
                p_text = f"p = {corr_stats['p_value']:.3g}"
                if corr_stats['p_value'] < 0.001:
                    p_text += " ***"
                elif corr_stats['p_value'] < 0.01:
                    p_text += " **"
                elif corr_stats['p_value'] < 0.05:
                    p_text += " *"
            else:
                spearman_text = "ρ = N/A"
                p_text = "p = N/A"

            # Add text boxes with simplified statistics (no CI, no sample size)
            ax.text(0.05, 0.95, spearman_text, transform=ax.transAxes, 
                   va='top', ha='left', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.5, pad=1, boxstyle="round,pad=0.3", edgecolor='none'))

            ax.text(0.05, 0.87, p_text, transform=ax.transAxes, 
                   va='top', ha='left', fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.5, pad=1, boxstyle="round,pad=0.3", edgecolor='none'))
            
            # Set consistent axis limits
            ax.set_xlim(data_ranges[var1])
            ax.set_ylim(data_ranges[var2])

            # Simplify x-axis ticks to show only min, 0, and max values with proper formatting
            x_min, x_max = data_ranges[var1]
            if x_min < 0 and x_max > 0:
                ax.set_xticks([x_min, 0, x_max])
            else:
                ax.set_xticks([x_min, x_max])
                
            # Format x-axis tick labels depending on variable type
            if var1 == 'Temp_Anomaly' or var1 == 'Prod_Anomaly':
                # Use integer format for temperature and production anomalies
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
            else:
                # Default formatting for other variables
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

            # Add thicker lines at x=0 and y=0 to highlight zero anomalies
            y_min, y_max = data_ranges[var2]
            if 0 >= y_min and 0 <= y_max:
                ax.axhline(y=0, color='gray', linewidth=1.5, linestyle='-', alpha=0.7, zorder=0)
            if 0 >= x_min and 0 <= x_max:
                ax.axvline(x=0, color='gray', linewidth=1.5, linestyle='-', alpha=0.7, zorder=0)

            # Make tick parameters larger
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            # Add axis labels to every plot
            # For y-axis, show the second variable in the correlation pair
            ax.set_ylabel(f"{ANOMALY_VARS[var2]['title']}\n({ANOMALY_VARS[var2]['unit']})", fontsize=12)
            
            # For x-axis, show the first variable in the correlation pair
            ax.set_xlabel(f"{ANOMALY_VARS[var1]['title']}\n({ANOMALY_VARS[var1]['unit']})", fontsize=12)
            
            # Only show x-axis label text for the bottom row (but keep ticks for all)
            if row < len(REGIONS) - 1:
                ax.xaxis.label.set_visible(False)
    
    # REMOVED: Legend creation for region types and years
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    # Save the figure
    output_file = os.path.join(OUTPUT_DIR, "all_regions_anomaly_correlations.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved combined correlation plot to {output_file}")
    
    # Close figure to free memory
    plt.close(fig)

def calculate_spearman_ci_fisher(x, y, alpha=0.05):
    """
    Calculate confidence interval for Spearman correlation using Fisher's z-transformation
    with correction factor for Spearman.
    
    Parameters:
    -----------
    x, y : arrays
        Data arrays to correlate
    alpha : float
        Significance level (default: 0.05 for 95% confidence)
        
    Returns:
    --------
    dict
        Dictionary with spearman correlation, p-value, and confidence intervals
    """
    # Clean data (remove NaN pairs)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = np.array(x)[mask]
    y_clean = np.array(y)[mask]
    
    n = len(x_clean)
    
    if n < 4:  # Need at least 4 points for this method (n-3 in denominator)
        if n >= 3:
            # Calculate regular Spearman without CI
            spearman_full, p_value = stats.spearmanr(x_clean, y_clean)
            return {
                'spearman': spearman_full,
                'p_value': p_value,
                'ci_lower': None,
                'ci_upper': None,
                'n': n
            }
        else:
            return {
                'spearman': None,
                'p_value': None,
                'ci_lower': None,
                'ci_upper': None,
                'n': n
            }
    
    # Calculate Spearman correlation
    rs, p_value = stats.spearmanr(x_clean, y_clean)
    
    # Fisher's z-transformation
    z_rs = np.arctanh(rs)
    
    # Standard error with correction factor for Spearman
    # Exactly as in the formula: σ²ξ = (1 + r²s/2)/(n-3)
    se = np.sqrt((1 + rs**2/2)/(n-3))
    
    # Normal quantile for alpha/2
    z_crit = stats.norm.ppf(1-alpha/2)  # This is positive
    
    # Calculate confidence bounds
    lower_z = z_rs - z_crit * se
    upper_z = z_rs + z_crit * se
    
    # Transform back
    lower_bound = np.tanh(lower_z)
    upper_bound = np.tanh(upper_z)
    
    return {
        'spearman': rs,
        'p_value': p_value,
        'ci_lower': lower_bound,
        'ci_upper': upper_bound,
        'n': n
    }

def create_summary_table(all_region_data):
    """
    Create a summary table of correlation statistics for all regions.
    
    Parameters:
    -----------
    all_region_data : dict
        Dictionary of region names and their dataframes
    """
    summary_data = []
    
    for region in REGIONS:
        df = all_region_data.get(region)
        if df is None or df.empty:
            continue
        
        for var1, var2 in CORRELATION_PAIRS:
            corr_stats = calculate_correlation(df, var1, var2)
            
            if corr_stats['spearman'] is None:
                continue
                
            summary_data.append({
                'Region': region.replace('_', ' ').title(),
                'Variables': f"{ANOMALY_VARS[var1]['title']} vs {ANOMALY_VARS[var2]['title']}",
                'Spearman': corr_stats['spearman'],
                'Linear_R': corr_stats['linear_reg']['r_value'] if corr_stats['linear_reg']['r_value'] is not None else np.nan,
                'Data_Points': len(df[[var1, var2]].dropna())
            })
    
    if not summary_data:
        print("No correlation data available for summary table")
        return
    
    # Convert to DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    output_file = os.path.join(OUTPUT_DIR, "regional_correlations_summary.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"Saved correlation summary to {output_file}")

def main():
    """Main function to execute regional correlation analysis with explicit CSV paths."""
    print("=" * 80)
    print("REGIONAL ANOMALY CORRELATION ANALYSIS - EXPLICIT CSV PATHS")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data for all regions using explicit CSV paths
    all_region_data = {}
    for region in REGIONS:
        print(f"\nProcessing region: {region}")
        df = load_region_data(region)
        if df is not None:
            all_region_data[region] = df
            print(f"  Data loaded successfully: {len(df)} records")
        else:
            print(f"  No data available for {region}")
    
    # Create combined plot
    create_combined_correlation_plot(all_region_data)
    
    # Create summary table
    create_summary_table(all_region_data)
    
    print("\nAll regional correlation analysis completed successfully!")

if __name__ == "__main__":
    main()