#!/usr/bin/env python3
"""
Script to extract rectangular regions from NetCDF ocean data files using corner coordinates.
Also calculates monthly climatologies from extracted regions.
"""
# %%
import os
import glob
import numpy as np
from netCDF4 import Dataset
import argparse
from datetime import datetime
import xarray as xr

# Define study regions as rectangles with corner coordinates
STUDY_REGIONS = [
    {
        'name': 'highnpp',
        'year': 2014,
        'season': '1',  # May-July
        'lon_min': 220, 'lon_max': 235,
        'lat_min': 45, 'lat_max': 55
    },
    {
        'name': 'highnpp',
        'year': 2014,
        'season': '2',  # August-October
        'lon_min': 220, 'lon_max': 235,
        'lat_min': 45, 'lat_max': 55
    },
    {
        'name': 'highnpp',
        'year': 2015,
        'season': '1',  # May-July
        'lon_min': 220, 'lon_max': 235,
        'lat_min': 45, 'lat_max': 55
    },
    {
        'name': 'highnpp',
        'year': 2015,
        'season': '2',  # August-October
        'lon_min': 220, 'lon_max': 235,
        'lat_min': 45, 'lat_max': 55
    },
    {
        'name': 'highnpp',
        'year': 2016,
        'season': '1',  # May-July
        'lon_min': 220, 'lon_max': 235,
        'lat_min': 45, 'lat_max': 55
    },
    {
        'name': 'highnpp',
        'year': 2016,
        'season': '2',  # August-October
        'lon_min': 220, 'lon_max': 235,
        'lat_min': 45, 'lat_max': 55
    },
    {
        'name': 'lownpp',
        'year': 2014,
        'season': '1',  # May-July
        'lon_min': 210, 'lon_max': 225,
        'lat_min': 40, 'lat_max': 50
    },
    {
        'name': 'lownpp_north',
        'year': 2014,
        'season': '2',  # August-October
        'lon_min': 210, 'lon_max': 225,
        'lat_min': 40, 'lat_max': 50
    },
    {
        'name': 'lownpp',
        'year': 2015,
        'season': '1',  # May-July
        'lon_min': 210, 'lon_max': 225,
        'lat_min': 40, 'lat_max': 50
    },
    {
        'name': 'lownpp_north',
        'year': 2015,
        'season': '2',  # August-October
        'lon_min': 210, 'lon_max': 225,
        'lat_min': 40, 'lat_max': 50
    },
    {
        'name': 'lownpp',
        'year': 2016,
        'season': '1',  # May-July
        'lon_min': 210, 'lon_max': 225,
        'lat_min': 40, 'lat_max': 50
    },
    {
        'name': 'lownpp_north',
        'year': 2016,
        'season': '2',  # August-October
        'lon_min': 210, 'lon_max': 225,
        'lat_min': 40, 'lat_max': 50
    },
]

def find_region_indices(lon_min, lon_max, lat_min, lat_max, lon_grid, lat_grid):
    """
    Find the indices corresponding to a rectangular region defined by corner coordinates.
    """
    # Check the range of longitude in the grid
    grid_lon_min = np.nanmin(lon_grid)
    grid_lon_max = np.nanmax(lon_grid)
    
    # Print more debugging info
    print(f"    Target region: Lon({lon_min}, {lon_max}), Lat({lat_min}, {lat_max})")
    print(f"    Grid range: Lon({grid_lon_min:.2f}, {grid_lon_max:.2f}), Lat({np.nanmin(lat_grid):.2f}, {np.nanmax(lat_grid):.2f})")
    
    # Adjust target longitudes for coordinate system compatibility
    target_lon_min = lon_min
    target_lon_max = lon_max
    
    # If grid is in 0-360 range (Pacific) and target is in -180 to 180 range
    if grid_lon_min >= 0 and grid_lon_max > 180 and lon_min < 0:
        target_lon_min = lon_min + 360
        target_lon_max = lon_max + 360
        print(f"    Converting coordinates: new target Lon({target_lon_min}, {target_lon_max})")
    
    # Create masks for each boundary
    mask_lon_min = lon_grid >= target_lon_min
    mask_lon_max = lon_grid <= target_lon_max
    mask_lat_min = lat_grid >= lat_min
    mask_lat_max = lat_grid <= lat_max
    
    # Combine masks to find points inside the rectangle
    region_mask = mask_lon_min & mask_lon_max & mask_lat_min & mask_lat_max
    
    # Find the indices of the region boundaries
    if not np.any(region_mask):
        print(f"WARNING: No grid points found within region boundaries.")
        # Return empty indices
        return 0, 0, 0, 0
    
    # Get i, j indices where the mask is True
    i_indices, j_indices = np.where(region_mask)
    
    if len(i_indices) == 0 or len(j_indices) == 0:
        return 0, 0, 0, 0
    
    # Get the min/max indices defining the rectangle
    i_min, i_max = np.min(i_indices), np.max(i_indices)
    j_min, j_max = np.min(j_indices), np.max(j_indices)
    
    # Print diagnostic info
    n_points = len(i_indices)
    region_width = j_max - j_min + 1
    region_height = i_max - i_min + 1
    
    print(f"    Found {n_points} grid points inside region")
    print(f"    Region size: {region_height}×{region_width} grid points")
    
    return i_min, i_max, j_min, j_max

def extract_region(ds, i_min, i_max, j_min, j_max):
    """
    Extract a rectangular region defined by indices.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset
    i_min, i_max : int
        Min/max i-indices (eta_rho)
    j_min, j_max : int
        Min/max j-indices (xi_rho)
    
    Returns:
    --------
    xarray.Dataset
        Dataset with extracted region
    """
    # Ensure indices are valid
    if i_min == i_max or j_min == j_max:
        print("  ERROR: Invalid region indices, cannot extract region")
        return None
    
    # Extract the region
    region = ds.isel(eta_rho=slice(i_min, i_max + 1), 
                    xi_rho=slice(j_min, j_max + 1))
    
    # Add attributes for the extracted region
    region.attrs['extraction_i_range'] = f"{i_min}:{i_max}"
    region.attrs['extraction_j_range'] = f"{j_min}:{j_max}"
    
    return region

def process_file(file_path, base_output_dir):
    """
    Process a single NetCDF file, extracting regions defined by corner coordinates.
    Skip regions that have already been processed.
    
    Parameters:
    -----------
    file_path : str
        Path to input NetCDF file
    base_output_dir : str
        Base directory for output files
    """
    # Extract base filename
    base_name = os.path.basename(file_path)
    print(f"Processing {base_name}...")
    
    # Check if all regions for this file already exist
    all_regions_exist = True
    for region in STUDY_REGIONS:
        region_output_dir = os.path.join(base_output_dir, region['name'])
        output_name_pattern = f"{os.path.splitext(base_name)[0]}_*x*_rect.nc"
        existing_files = glob.glob(os.path.join(region_output_dir, output_name_pattern))
        if not existing_files:
            all_regions_exist = False
            break
    
    # Skip if all regions already exist
    if all_regions_exist:
        print(f"  Skipping {base_name} - all regions already processed")
        return
    
    # Open dataset with xarray - only load coordinates first to save memory
    ds = xr.open_dataset(file_path)
    
    # Extract lon and lat grids
    lon_grid = ds.lon_rho.values
    lat_grid = ds.lat_rho.values
    
    # Check if coordinates are valid
    if np.isnan(lon_grid).all() or np.isnan(lat_grid).all():
        print(f"  ERROR: Coordinate data contains only NaN values in {base_name}")
        print(f"  Check that 'lon_rho' and 'lat_rho' variables exist and contain valid data")
        # Print available variables to help diagnose the issue
        print(f"  Available variables in file: {list(ds.variables)}")
        return
    
    # Print coordinate range information to help with debugging
    print(f"  Grid longitude range: {np.nanmin(lon_grid):.2f} to {np.nanmax(lon_grid):.2f}")
    print(f"  Grid latitude range: {np.nanmin(lat_grid):.2f} to {np.nanmax(lat_grid):.2f}")
    
    # Process each study region
    for region in STUDY_REGIONS:
        print(f"  Processing region: {region['name']}")
        
        # Find indices for the region corners
        i_min, i_max, j_min, j_max = find_region_indices(
            region['lon_min'], region['lon_max'], 
            region['lat_min'], region['lat_max'],
            lon_grid, lat_grid
        )
        
        if i_min == i_max or j_min == j_max:
            print(f"  Skipping region {region['name']} - invalid indices")
            continue
        
        # Calculate region dimensions
        region_width = j_max - j_min + 1
        region_height = i_max - i_min + 1
        
        # Create region-specific output directory
        region_output_dir = os.path.join(base_output_dir, region['name'])
        os.makedirs(region_output_dir, exist_ok=True)
        
        # Construct output filename
        output_name = f"{os.path.splitext(base_name)[0]}_{region_width}x{region_height}_rect.nc"
        output_path = os.path.join(region_output_dir, output_name)
        
        # Check if output file already exists
        if os.path.exists(output_path):
            print(f"  Skipping region {region['name']} - output file already exists: {output_name}")
            continue
        
        # Extract region
        extracted = extract_region(ds, i_min, i_max, j_min, j_max)
        
        if extracted is None:
            print(f"  Failed to extract region {region['name']}")
            continue
        
        # Calculate the actual corner coordinates of the extracted region
        actual_lon_min = float(np.min(lon_grid[i_min:i_max+1, j_min:j_max+1]))
        actual_lon_max = float(np.max(lon_grid[i_min:i_max+1, j_min:j_max+1]))
        actual_lat_min = float(np.min(lat_grid[i_min:i_max+1, j_min:j_max+1]))
        actual_lat_max = float(np.max(lat_grid[i_min:i_max+1, j_min:j_max+1]))
        
        # Add metadata about extraction
        extracted.attrs['region_name'] = region['name']
        extracted.attrs['region_target_lon_min'] = region['lon_min']
        extracted.attrs['region_target_lon_max'] = region['lon_max']
        extracted.attrs['region_target_lat_min'] = region['lat_min']
        extracted.attrs['region_target_lat_max'] = region['lat_max']
        extracted.attrs['region_actual_lon_min'] = actual_lon_min
        extracted.attrs['region_actual_lon_max'] = actual_lon_max
        extracted.attrs['region_actual_lat_min'] = actual_lat_min
        extracted.attrs['region_actual_lat_max'] = actual_lat_max
        extracted.attrs['extraction_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        extracted.attrs['source_file'] = file_path
        extracted.attrs['region_width'] = region_width
        extracted.attrs['region_height'] = region_height
        
        # Save extracted region (with compression to save space)
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in extracted.data_vars}
        extracted.to_netcdf(output_path, encoding=encoding)
        print(f"  Saved {output_path}")

def calculate_monthly_climatologies(base_output_dir, year_range='2011-2021'):
    """
    Calculate monthly climatologies from extracted regions.
    Skip climatologies that have already been calculated.
    
    Parameters:
    -----------
    base_output_dir : str
        Base directory containing the region subdirectories
    year_range : str
        Range of years to include in climatology (format: 'YYYY-YYYY')
    """
    print(f"\nCalculating monthly climatologies for {year_range}...")
    
    # Parse year range
    start_year, end_year = map(int, year_range.split('-'))
    years = range(start_year, end_year + 1)
    
    # Process each study region
    for region in STUDY_REGIONS:
        region_dir = os.path.join(base_output_dir, region['name'])
        if not os.path.exists(region_dir):
            print(f"  Warning: Directory for {region['name']} not found at {region_dir}")
            continue
            
        print(f"\nProcessing climatologies for {region['name']}...")
        
        # Create climatology directory
        clim_dir = os.path.join(region_dir, 'climatology')
        os.makedirs(clim_dir, exist_ok=True)
        
        # Process each month
        for month in range(1, 13):
            month_str = f"{month:03d}"
            print(f"  Processing month {month_str}...")
            
            # Find all files for this month across years
            monthly_files = []
            for year in years:
                pattern = f"z_avg_{year}_{month_str}_*_rect.nc"
                files = glob.glob(os.path.join(region_dir, pattern))
                monthly_files.extend(files)
            
            if not monthly_files:
                print(f"    No files found for month {month_str}, skipping")
                continue
                
            print(f"    Found {len(monthly_files)} files for month {month_str}")
            
            # Extract region dimensions from one of the files
            file_parts = os.path.basename(monthly_files[0]).split('_')
            region_dims = next((part for part in file_parts if 'x' in part and part.endswith('rect.nc')), "")
            region_dims = region_dims.replace('rect.nc', '')
            
            # Check if climatology file already exists
            output_path = os.path.join(clim_dir, f"climatology_month{month_str}_{region_dims}_rect.nc")
            if os.path.exists(output_path):
                print(f"    Skipping month {month_str} - climatology already exists")
                continue
            
            # Use xarray to load and average the datasets
            try:
                # Load all datasets for this month
                datasets = [xr.open_dataset(file) for file in monthly_files]
                
                # Combine datasets along a new "year" dimension
                combined = xr.concat(datasets, dim="year")
                
                # Calculate mean across years for each variable at each coordinate and depth
                climatology = combined.mean(dim="year")
                
                # Add metadata
                climatology.attrs['climatology_type'] = 'monthly'
                climatology.attrs['climatology_month'] = month
                climatology.attrs['climatology_years'] = year_range
                climatology.attrs['files_included'] = len(monthly_files)
                climatology.attrs['generation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Apply compression
                comp = dict(zlib=True, complevel=5)
                encoding = {var: comp for var in climatology.data_vars}
                
                climatology.to_netcdf(output_path, encoding=encoding)
                print(f"    Saved climatology to {output_path}")
                
                # Close datasets to free memory
                for ds in datasets:
                    ds.close()
                    
            except Exception as e:
                print(f"    Error calculating climatology for month {month_str}: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract rectangular regions from NetCDF files using corner coordinates')
    parser.add_argument('--input_dir', default='/nfs/sea/work/bblaser/z_avg_meanpool_domain',
                       help='Directory containing input NetCDF files')
    parser.add_argument('--output_dir', default='/nfs/sea/work/bblaser/regions2',
                       help='Base directory for output files (region-specific subdirectories will be created)')
    parser.add_argument('--pattern', default='z_avg_*_37zlevs_full_1x1meanpool_downsampling.nc',
                       help='Glob pattern for input files')
    parser.add_argument('--year_range', default='2011-2021',
                       help='Range of years to process (e.g., 2011-2021)')
    parser.add_argument('--drop_nan_vars', action='store_true',
                       help='Drop variables that contain only NaN values')
    parser.add_argument('--calculate_climatology', action='store_true',
                       help='Calculate monthly climatology after processing files')
    parser.add_argument('--climatology_only', action='store_true',
                       help='Skip file processing and only calculate climatology from existing files')
    args = parser.parse_known_args()[0]
    
    # Create base output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse year range
    start_year, end_year = map(int, args.year_range.split('-'))
    
    # Process files if not in climatology-only mode
    if not args.climatology_only:
        # Get list of files matching pattern
        file_pattern = os.path.join(args.input_dir, args.pattern)
        all_files = sorted(glob.glob(file_pattern))
        
        # Filter files by year range
        filtered_files = []
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            try:
                # Extract year from filename (assuming pattern z_avg_YYYY_*)
                year = int(file_name.split('_')[2])
                if start_year <= year <= end_year:
                    filtered_files.append(file_path)
            except (IndexError, ValueError):
                print(f"Warning: Could not extract year from {file_name}, skipping")
        
        print(f"Found {len(filtered_files)} files to process between {start_year}-{end_year}")
        
        # Process each file
        for file_path in filtered_files:
            try:
                process_file(file_path, args.output_dir)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Calculate climatologies if requested
    if args.calculate_climatology or args.climatology_only:
        calculate_monthly_climatologies(args.output_dir, args.year_range)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
# %%