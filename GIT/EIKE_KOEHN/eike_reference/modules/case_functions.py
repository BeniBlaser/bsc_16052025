"""
functions that are repeatedly used for extracting information
"""

def get_z_dz_lon_lat_and_area(params):
    print('Get dz (vertical thickness of layers).')
    #dummy_year = params.hindcast_analysis_start_year
    #regridded_model_file = [s for s in params._fname_model_output_regridded() if str(dummy_year) in s][0]
    #dummy_file = xr.open_dataset(params.dir_roms_only_present_raw+regridded_model_file)
    dummy_file = xr.open_dataset(params.dir_roms_only_present_raw+params._fname_model_output_raw()[0])
    z = 's levels' #dummy_file.depth.values
    dz = 's levels' #dummy_file.dz.values
    lon = dummy_file.lon_rho.values
    lat = dummy_file.lat_rho.values
    area = 1/dummy_file.pm.values * 1/dummy_file.pn.values
    mask = dummy_file.mask_rho.values
    return z, dz, lon, lat, area,mask

def get_pm_pn(params):
    #dummy_year = params.hindcast_analysis_start_year
    #regridded_model_file = [s for s in params._fname_model_output_regridded() if str(dummy_year) in s][0]
    #dummy_file = xr.open_dataset(params.dir_roms_only_present_raw+regridded_model_file)
    dummy_file = xr.open_dataset(params.dir_roms_only_present_raw+params._fname_model_output_raw()[0])
    pm = dummy_file.pm.values
    pn = dummy_file.pn.values
    return pm, pn