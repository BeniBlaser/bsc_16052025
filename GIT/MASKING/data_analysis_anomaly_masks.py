#!/usr/bin/env python3
"""
Correlation analysis between temperature, production, and POC flux anomalies.
Creates scatter plots with linear regression and Spearman rank correlation coefficients
with Fisher z-transformation confidence intervals for different threshold datasets.


Author: Beni Blaser  
This script was developed with assistance from Claude Sonnet 3.7 (via Copilot), which supported code creation, debugging, and documentation.
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from scipy import stats
import seaborn as sns
import cmocean as cmo

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

# Define color palettes using CMasher colormaps
CM_COLORS = {
    'whole_region': {
        '2014': plt.get_cmap('cmo.gray')(0.3),
        '2015': plt.get_cmap('cmo.gray')(0.5),
        '2016': plt.get_cmap('cmo.gray')(0.7)
    },
    10: {
        '2014': plt.get_cmap('cmo.dense')(0.3),
        '2015': plt.get_cmap('cmo.dense')(0.5),
        '2016': plt.get_cmap('cmo.dense')(0.7)
    },
    30: {
        '2014': plt.get_cmap('cmo.tempo')(0.3),
        '2015': plt.get_cmap('cmo.tempo')(0.5),
        '2016': plt.get_cmap('cmo.tempo')(0.7)
    },
    50: {
        '2014': plt.get_cmap('cmo.matter')(0.3),
        '2015': plt.get_cmap('cmo.matter')(0.5),
        '2016': plt.get_cmap('cmo.matter')(0.7)
    }
}

# Blob period definition
BLOB_START = datetime(2014, 5, 1)
BLOB_END = datetime(2016, 10, 31)

def get_color_by_year(year, dataset_key):
    """Return color based on year and dataset using CMasher colormaps."""
    year_str = str(year)
    
    if dataset_key == 'whole_region':
        palette = CM_COLORS['whole_region']
    else:
        threshold = int(dataset_key.split('_')[1])
        palette = CM_COLORS[threshold]
    
    return palette.get(year_str, plt.get_cmap('cmo.gray')(0.5))

def calculate_spearman_ci_fisher(x, y, alpha=0.05):
    """
    Calculate confidence interval for Spearman correlation using Fisher's z-transformation
    with correction factor for Spearman, exactly as in the formula:
    
    tanh(atanh(rs) ± √[(1 + rs²/2)/(n-3)] * z_α/2)
    
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

def prepare_correlation_data(data_dict):
    """
    Prepare data for correlation analysis by matching dates between variables.
    Only include data from 2014-2016 (Blob period).
    """
    paired_data = {
        'whole_region': {
            'temp_vs_TOT_PROD': {'temp': [], 'TOT_PROD': [], 'dates': []},
            'temp_vs_POC_FLUX_IN': {'temp': [], 'POC_FLUX_IN': [], 'dates': []},
            'TOT_PROD_vs_POC_FLUX_IN': {'TOT_PROD': [], 'POC_FLUX_IN': [], 'dates': []}
        }
    }
    
    # Initialize threshold data structures
    for threshold in THRESHOLDS:
        paired_data[f'threshold_{threshold}'] = {
            'temp_vs_TOT_PROD': {'temp': [], 'TOT_PROD': [], 'dates': []},
            'temp_vs_POC_FLUX_IN': {'temp': [], 'POC_FLUX_IN': [], 'dates': []},
            'TOT_PROD_vs_POC_FLUX_IN': {'TOT_PROD': [], 'POC_FLUX_IN': [], 'dates': []}
        }
    
    # Process whole region data
    temp_wr = data_dict.get('temp_whole_region', pd.DataFrame())
    prod_wr = data_dict.get('TOT_PROD_whole_region', pd.DataFrame())
    poc_wr = data_dict.get('POC_FLUX_IN_whole_region', pd.DataFrame())
    
    # Filter for Blob period (2014-2016)
    if not temp_wr.empty:
        temp_wr = temp_wr[(temp_wr['date'].dt.year >= 2014) & (temp_wr['date'].dt.year <= 2016)]
    if not prod_wr.empty:
        prod_wr = prod_wr[(prod_wr['date'].dt.year >= 2014) & (prod_wr['date'].dt.year <= 2016)]
    if not poc_wr.empty:
        poc_wr = poc_wr[(poc_wr['date'].dt.year >= 2014) & (poc_wr['date'].dt.year <= 2016)]
    
    if not temp_wr.empty and not prod_wr.empty:
        # Create dictionaries for fast lookup
        temp_dict = {row['date']: row['anomaly'] for _, row in temp_wr.iterrows() if 'date' in row and 'anomaly' in row}
        prod_dict = {row['date']: row['anomaly'] for _, row in prod_wr.iterrows() if 'date' in row and 'anomaly' in row}
        
        # Find common dates and extract paired values
        common_dates = set(temp_dict.keys()) & set(prod_dict.keys())
        for date in sorted(common_dates):
            paired_data['whole_region']['temp_vs_TOT_PROD']['temp'].append(temp_dict[date])
            paired_data['whole_region']['temp_vs_TOT_PROD']['TOT_PROD'].append(prod_dict[date])
            paired_data['whole_region']['temp_vs_TOT_PROD']['dates'].append(date)
    
    # Similar processing for other variable pairs and thresholds
    if not temp_wr.empty and not poc_wr.empty:
        temp_dict = {row['date']: row['anomaly'] for _, row in temp_wr.iterrows() if 'date' in row and 'anomaly' in row}
        poc_dict = {row['date']: row['anomaly'] for _, row in poc_wr.iterrows() if 'date' in row and 'anomaly' in row}
        
        common_dates = set(temp_dict.keys()) & set(poc_dict.keys())
        for date in sorted(common_dates):
            paired_data['whole_region']['temp_vs_POC_FLUX_IN']['temp'].append(temp_dict[date])
            paired_data['whole_region']['temp_vs_POC_FLUX_IN']['POC_FLUX_IN'].append(poc_dict[date])
            paired_data['whole_region']['temp_vs_POC_FLUX_IN']['dates'].append(date)
    
    if not prod_wr.empty and not poc_wr.empty:
        prod_dict = {row['date']: row['anomaly'] for _, row in prod_wr.iterrows() if 'date' in row and 'anomaly' in row}
        poc_dict = {row['date']: row['anomaly'] for _, row in poc_wr.iterrows() if 'date' in row and 'anomaly' in row}
        
        common_dates = set(prod_dict.keys()) & set(poc_dict.keys())
        for date in sorted(common_dates):
            paired_data['whole_region']['TOT_PROD_vs_POC_FLUX_IN']['TOT_PROD'].append(prod_dict[date])
            paired_data['whole_region']['TOT_PROD_vs_POC_FLUX_IN']['POC_FLUX_IN'].append(poc_dict[date])
            paired_data['whole_region']['TOT_PROD_vs_POC_FLUX_IN']['dates'].append(date)
    
    # Process threshold data
    for threshold in THRESHOLDS:
        temp_th = data_dict.get(f'temp_threshold_{threshold}', pd.DataFrame())
        prod_th = data_dict.get(f'TOT_PROD_threshold_{threshold}', pd.DataFrame())
        poc_th = data_dict.get(f'POC_FLUX_IN_threshold_{threshold}', pd.DataFrame())
        
        # Process temp vs TOT_PROD
        if not temp_th.empty and not prod_th.empty:
            temp_dict = {row['date']: row['anomaly'] for _, row in temp_th.iterrows() if 'date' in row and 'anomaly' in row}
            prod_dict = {row['date']: row['anomaly'] for _, row in prod_th.iterrows() if 'date' in row and 'anomaly' in row}
            
            common_dates = set(temp_dict.keys()) & set(prod_dict.keys())
            for date in sorted(common_dates):
                paired_data[f'threshold_{threshold}']['temp_vs_TOT_PROD']['temp'].append(temp_dict[date])
                paired_data[f'threshold_{threshold}']['temp_vs_TOT_PROD']['TOT_PROD'].append(prod_dict[date])
                paired_data[f'threshold_{threshold}']['temp_vs_TOT_PROD']['dates'].append(date)
        
        # Process temp vs POC_FLUX_IN
        if not temp_th.empty and not poc_th.empty:
            temp_dict = {row['date']: row['anomaly'] for _, row in temp_th.iterrows() if 'date' in row and 'anomaly' in row}
            poc_dict = {row['date']: row['anomaly'] for _, row in poc_th.iterrows() if 'date' in row and 'anomaly' in row}
            
            common_dates = set(temp_dict.keys()) & set(poc_dict.keys())
            for date in sorted(common_dates):
                paired_data[f'threshold_{threshold}']['temp_vs_POC_FLUX_IN']['temp'].append(temp_dict[date])
                paired_data[f'threshold_{threshold}']['temp_vs_POC_FLUX_IN']['POC_FLUX_IN'].append(poc_dict[date])
                paired_data[f'threshold_{threshold}']['temp_vs_POC_FLUX_IN']['dates'].append(date)
        
        # Process TOT_PROD vs POC_FLUX_IN
        if not prod_th.empty and not poc_th.empty:
            prod_dict = {row['date']: row['anomaly'] for _, row in prod_th.iterrows() if 'date' in row and 'anomaly' in row}
            poc_dict = {row['date']: row['anomaly'] for _, row in poc_th.iterrows() if 'date' in row and 'anomaly' in row}
            
            common_dates = set(prod_dict.keys()) & set(poc_dict.keys())
            for date in sorted(common_dates):
                paired_data[f'threshold_{threshold}']['TOT_PROD_vs_POC_FLUX_IN']['TOT_PROD'].append(prod_dict[date])
                paired_data[f'threshold_{threshold}']['TOT_PROD_vs_POC_FLUX_IN']['POC_FLUX_IN'].append(poc_dict[date])
                paired_data[f'threshold_{threshold}']['TOT_PROD_vs_POC_FLUX_IN']['dates'].append(date)
    
    return paired_data

def calculate_correlations(paired_data):
    """
    Calculate Spearman rank correlation with Fisher z-transformation confidence intervals.
    """
    correlations = {}
    
    for dataset_key, dataset in paired_data.items():
        correlations[dataset_key] = {}
        
        for pair_key, pair_data in dataset.items():
            # Extract variable names
            vars = pair_key.split('_vs_')
            
            # Skip if we don't have enough data
            if len(pair_data[vars[0]]) < 3 or len(pair_data[vars[1]]) < 3:
                correlations[dataset_key][pair_key] = {
                    'spearman': None,
                    'p_value': None,
                    'ci_lower': None,
                    'ci_upper': None,
                    'n': 0,
                    'slope': None, 
                    'intercept': None
                }
                continue
            
            # Calculate Spearman correlation with Fisher z-transform CI
            spearman_results = calculate_spearman_ci_fisher(
                np.array(pair_data[vars[0]]), 
                np.array(pair_data[vars[1]])
            )
            
            # Calculate slope and intercept for visualization
            x = np.array(pair_data[vars[0]])
            y = np.array(pair_data[vars[1]])
            
            # Handle NaN values
            mask = ~np.isnan(x) & ~np.isnan(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) >= 2:  # Need at least 2 points for regression
                slope, intercept, _, _, _ = stats.linregress(x_clean, y_clean)
            else:
                slope, intercept = None, None
            
            correlations[dataset_key][pair_key] = {
                'spearman': spearman_results['spearman'],
                'p_value': spearman_results['p_value'],
                'ci_lower': spearman_results['ci_lower'],
                'ci_upper': spearman_results['ci_upper'],
                'n': spearman_results['n'],
                'slope': slope,
                'intercept': intercept
            }
    
    return correlations

def create_correlation_plots(paired_data, correlations):
    """
    Create scatter plots showing correlation data with quadrant percentages.
    """
    # Define pair titles
    pair_titles = {
        'temp_vs_TOT_PROD': 'Temperature vs. NPP Anomalies',
        'temp_vs_POC_FLUX_IN': 'Temperature vs. POC Flux Anomalies',
        'TOT_PROD_vs_POC_FLUX_IN': 'NPP vs. POC Flux Anomalies'
    }
    
    # Define dataset titles
    dataset_titles = {
        'whole_region': 'Whole Region',
        'threshold_10': '10% Threshold',
        'threshold_30': '30% Threshold',
        'threshold_50': '50% Threshold'
    }
    
    # Create a figure for each variable pair
    for pair_key in ['temp_vs_TOT_PROD', 'temp_vs_POC_FLUX_IN', 'TOT_PROD_vs_POC_FLUX_IN']:
        # Extract variable names
        vars = pair_key.split('_vs_')
        var1, var2 = vars[0], vars[1]
        
        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(pair_titles[pair_key], fontsize=16)
        axes = axes.flatten()
        
        # Track min/max for consistent axes
        all_x_values = []
        all_y_values = []
        
        # Collect all data points to determine axis limits
        for dataset_key in ['whole_region'] + [f'threshold_{t}' for t in THRESHOLDS]:
            if dataset_key in paired_data and pair_key in paired_data[dataset_key]:
                x_data = paired_data[dataset_key][pair_key][var1]
                y_data = paired_data[dataset_key][pair_key][var2]
                all_x_values.extend(x_data)
                all_y_values.extend(y_data)
        
        # Determine axis limits with padding
        if all_x_values and all_y_values:
            x_min, x_max = min(all_x_values), max(all_x_values)
            y_min, y_max = min(all_y_values), max(all_y_values)
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min = x_min - 0.1 * x_range
            x_max = x_max + 0.1 * x_range
            y_min = y_min - 0.1 * y_range
            y_max = y_max + 0.1 * y_range
        else:
            x_min, x_max = -1, 1
            y_min, y_max = -1, 1
        
        # Plot each dataset
        for i, dataset_key in enumerate(['whole_region'] + [f'threshold_{t}' for t in THRESHOLDS]):
            ax = axes[i]
            
            if dataset_key in paired_data and pair_key in paired_data[dataset_key]:
                dataset = paired_data[dataset_key][pair_key]
                
                # Skip if not enough data
                if len(dataset[var1]) < 2 or len(dataset[var2]) < 2:
                    ax.text(0.5, 0.5, "Insufficient data", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(dataset_titles[dataset_key])
                    continue
                
                # Get correlation data
                corr_data = correlations[dataset_key][pair_key]
                
                # Extract x and y data
                x_data = np.array(dataset[var1])
                y_data = np.array(dataset[var2])
                dates = dataset['dates']
                
                # Handle NaN values
                mask = ~np.isnan(x_data) & ~np.isnan(y_data)
                x_clean = x_data[mask]
                y_clean = y_data[mask]
                dates_clean = [date for idx, date in enumerate(dates) if mask[idx]]
                
                # Plot scatter points with year-based colors
                for j, (x, y, date) in enumerate(zip(x_clean, y_clean, dates_clean)):
                    year = date.year
                    color = get_color_by_year(year, dataset_key)
                    ax.scatter(x, y, color=color, alpha=0.7, s=35, edgecolor='black')
                
                # Calculate percentage of points in each quadrant
                if len(x_clean) > 0:
                    total_points = len(x_clean)
                    q1 = np.sum((x_clean > 0) & (y_clean > 0))  # top right
                    q2 = np.sum((x_clean < 0) & (y_clean > 0))  # top left
                    q3 = np.sum((x_clean < 0) & (y_clean < 0))  # bottom left
                    q4 = np.sum((x_clean > 0) & (y_clean < 0))  # bottom right
                    
                    # Calculate percentages
                    q1_pct = 100 * q1 / total_points if total_points > 0 else 0
                    q2_pct = 100 * q2 / total_points if total_points > 0 else 0
                    q3_pct = 100 * q3 / total_points if total_points > 0 else 0
                    q4_pct = 100 * q4 / total_points if total_points > 0 else 0
                    
                    # Add percentage labels in each quadrant
                    if q1 > 0:  # Top right
                        ax.text(x_min + 0.75*(x_max-x_min), y_min + 0.75*(y_max-y_min), 
                                f"{q1_pct:.0f}%", ha='center', va='center', 
                                fontsize=11, fontweight='bold', color='black',
                                bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.2", edgecolor='none'))
                    
                    if q2 > 0:  # Top left
                        ax.text(x_min + 0.25*(x_max-x_min), y_min + 0.75*(y_max-y_min), 
                                f"{q2_pct:.0f}%", ha='center', va='center', 
                                fontsize=11, fontweight='bold', color='black',
                                bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.2", edgecolor='none'))
                    
                    if q3 > 0:  # Bottom left
                        ax.text(x_min + 0.25*(x_max-x_min), y_min + 0.25*(y_max-y_min), 
                                f"{q3_pct:.0f}%", ha='center', va='center', 
                                fontsize=11, fontweight='bold', color='black',
                                bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.2", edgecolor='none'))
                    
                    if q4 > 0:  # Bottom right
                        ax.text(x_min + 0.75*(x_max-x_min), y_min + 0.25*(y_max-y_min), 
                                f"{q4_pct:.0f}%", ha='center', va='center', 
                                fontsize=11, fontweight='bold', color='black',
                                bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.2", edgecolor='none'))

                # Add correlation info - only Spearman and p-value (NO CI or sample size)
                if corr_data['spearman'] is not None:
                    spearman_text = f"ρ = {corr_data['spearman']:.3f}"
                    
                    # Add p-value with significance stars
                    p_text = f"p = {corr_data['p_value']:.3g}"
                    if corr_data['p_value'] < 0.001:
                        p_text += " ***"
                    elif corr_data['p_value'] < 0.01:
                        p_text += " **"
                    elif corr_data['p_value'] < 0.05:
                        p_text += " *"
                else:
                    spearman_text = "ρ = N/A"
                    p_text = "p = N/A"

                # Display ONLY Spearman and p-value
                ax.text(0.05, 0.95, spearman_text, transform=ax.transAxes, 
                       va='top', ha='left', fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle="round,pad=0.3", edgecolor='none'))

                ax.text(0.05, 0.87, p_text, transform=ax.transAxes, 
                       va='top', ha='left', fontsize=9,
                       bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle="round,pad=0.3", edgecolor='none'))
                
                # Set consistent axis limits
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                
                # Add PRONOUNCED zero lines at x=0 and y=0
                ax.axhline(y=0, color='black', linewidth=2.0, linestyle='-', alpha=0.8, zorder=1)
                ax.axvline(x=0, color='black', linewidth=2.0, linestyle='-', alpha=0.8, zorder=1)
                
                # Add legend for color years
                handles = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=get_color_by_year(2014, dataset_key), 
                              label='2014', markersize=8, markeredgecolor='black'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=get_color_by_year(2015, dataset_key), 
                              label='2015', markersize=8, markeredgecolor='black'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=get_color_by_year(2016, dataset_key), 
                              label='2016', markersize=8, markeredgecolor='black')
                ]
                ax.legend(handles=handles, loc='lower right', fontsize=8, framealpha=0.7)
            else:
                ax.text(0.5, 0.5, "No data available", 
                       ha='center', va='center', transform=ax.transAxes)
            
            # Set title and labels
            ax.set_title(dataset_titles[dataset_key])
            ax.set_xlabel(f"{TITLES[var1]} Anomaly ({UNITS[var1]})")
            ax.set_ylabel(f"{TITLES[var2]} Anomaly ({UNITS[var2]})")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        output_file = os.path.join(OUTPUT_DIR, f"correlation_{var1}_vs_{var2}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved correlation plot to {output_file}")

def create_correlation_summary(correlations):
    """Create a summary table of all correlation values with confidence intervals and sample sizes."""
    summary_data = []
    for dataset_key in ['whole_region'] + [f'threshold_{t}' for t in THRESHOLDS]:
        if dataset_key in correlations:
            for pair_key in ['temp_vs_TOT_PROD', 'temp_vs_POC_FLUX_IN', 'TOT_PROD_vs_POC_FLUX_IN']:
                if pair_key in correlations[dataset_key]:
                    corr_data = correlations[dataset_key][pair_key]
                    
                    # Format values
                    spearman = f"{corr_data['spearman']:.3f}" if corr_data['spearman'] is not None else "N/A"
                    
                    # Format CI
                    ci = f"[{corr_data['ci_lower']:.3f}, {corr_data['ci_upper']:.3f}]" if (
                        corr_data['ci_lower'] is not None and 
                        corr_data['ci_upper'] is not None
                    ) else "N/A"
                    
                    # Format p-value with significance
                    if corr_data['p_value'] is not None:
                        p_value = f"{corr_data['p_value']:.5f}"
                        if corr_data['p_value'] < 0.001:
                            p_value += " ***"
                        elif corr_data['p_value'] < 0.01:
                            p_value += " **"
                        elif corr_data['p_value'] < 0.05:
                            p_value += " *"
                    else:
                        p_value = "N/A"
                    
                    # Format sample size
                    n = str(corr_data['n']) if corr_data['n'] is not None else "0"
                    
                    # Format dataset and variable names
                    if dataset_key == 'whole_region':
                        dataset_name = "Whole Region"
                    else:
                        threshold = dataset_key.split('_')[1]
                        dataset_name = f"{threshold}% Threshold"
                    
                    vars = pair_key.split('_vs_')
                    pair_name = f"{TITLES[vars[0]]} vs {TITLES[vars[1]]}"
                    
                    summary_data.append([dataset_name, pair_name, spearman, ci, p_value, n])
    
    # Create CSV summary
    output_file = os.path.join(OUTPUT_DIR, "anomaly_correlations_summary.csv")
    with open(output_file, 'w') as f:
        f.write("Dataset,Variable Pair,Spearman Correlation,95% Confidence Interval,p-value,Sample Size\n")
        for row in summary_data:
            f.write(",".join(row) + "\n")
    
    print(f"Saved correlation summary to {output_file}")

def print_data_point_counts(paired_data):
    """Print the number of data points for each correlation."""
    print("\nData point counts for each correlation:")
    print("-" * 50)
    
    for dataset_key in ['whole_region'] + [f'threshold_{t}' for t in THRESHOLDS]:
        if dataset_key in paired_data:
            print(f"\n{dataset_key.replace('_', ' ').title()}:")
            for pair_key in ['temp_vs_TOT_PROD', 'temp_vs_POC_FLUX_IN', 'TOT_PROD_vs_POC_FLUX_IN']:
                if pair_key in paired_data[dataset_key]:
                    vars = pair_key.split('_vs_')
                    point_count = len(paired_data[dataset_key][pair_key][vars[0]])
                    print(f"  {pair_key.replace('_vs_', ' vs. ').replace('_', ' ').title()}: {point_count} points")
                else:
                    print(f"  {pair_key.replace('_vs_', ' vs. ').replace('_', ' ').title()}: No data")

def main():
    """Main function to execute the correlation analysis workflow."""
    print("=" * 80)
    print("ANOMALIES CORRELATION ANALYSIS WITH FISHER CI FOR SPEARMAN")
    print("=" * 80)
    
    # Step 1: Load all data
    all_data = load_data()
    
    if not all_data:
        print("Error: No data available. Check data directory and file patterns.")
        return
    
    # Step 2: Prepare data for correlation analysis
    paired_data = prepare_correlation_data(all_data)
    
    # Print data point counts
    print_data_point_counts(paired_data)
    
    # Step 3: Calculate correlations with confidence intervals
    correlations = calculate_correlations(paired_data)
    
    # Step 4: Create correlation plots with proper confidence intervals
    create_correlation_plots(paired_data, correlations)
    
    print("\nAll correlation analysis completed successfully!")

if __name__ == "__main__":
    main()
