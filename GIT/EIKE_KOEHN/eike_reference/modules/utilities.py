#%%
"""
utilities for processing the model outputs
author: Eik E. Koehn
date: Dec 8, 2023
"""

#%% LOAD PACKAGES
import xarray as xr
import glob

#%% DEfINE FUNCTIONS

# define directory and filename for year, modelsetup and scenario
def get_list_of_combined_fdirs_and_fnames_raw_for_specific_year(params,model_config,scenario,year):
    fcombs = []
    if model_config == 'roms_only':
        if scenario == 'present':
            fdir = params.dir_roms_only_present_raw
        elif scenario == 'ssp245':
            fdir = params.dir_roms_only_ssp245_raw
        elif scenario == 'ssp585':
            fdir = params.dir_roms_only_ssp585_raw
        fname = "{}_{}_avg.nc".format(params.grid_name,year)
        fcombs.append('{}{}'.format(fdir,fname))
    elif model_config == 'romsoc_fully_coupled':
        if scenario == 'present':
            fdir = params.dir_romsoc_fully_coupled_present_raw.replace('YYYY',str(year))
        elif scenario == 'ssp245':
            fdir = params.dir_romsoc_fully_coupled_ssp245_raw.replace('YYYY',str(year))
        elif scenario == 'ssp585':
            fdir = params.dir_romsoc_fully_coupled_ssp585_raw.replace('YYYY',str(year))
        for i in range(1,13):
            fcombs.append("{}avg_{:03d}.nc".format(fdir,i))
    return fcombs

def set_fdir_and_fname_raw_for_monthly_spinup_data(params,model_config,scenario,year):
    if model_config == 'roms_only':
        if scenario == 'present':
            fdir = params.dir_roms_only_present_spinup
        elif scenario == 'ssp245':
            fdir = params.dir_roms_only_ssp245_spinup
        elif scenario == 'ssp585':
            fdir = params.dir_roms_only_ssp585_spinup
    elif model_config == 'romsoc_fully_coupled':
        if scenario == 'present':
            fdir = params.dir_romsoc_fully_coupled_present_spinup
        elif scenario == 'ssp245':
            fdir = params.dir_romsoc_fully_coupled_ssp245_spinup
        elif scenario == 'ssp585':
            fdir = params.dir_romsoc_fully_coupled_ssp585_spinup
    fname = "{}_{}_avg.nc".format(params.grid_name,year)
    fcomb = '{}{}'.format(fdir,fname)
    return fcomb, fdir, fname

def set_fdir_and_fname_raw_for_monthly_data(params,model_config,scenario,year):
    if model_config == 'roms_only':
        if scenario == 'present':
            fdir = params.dir_roms_only_present_monthly
        elif scenario == 'ssp245':
            fdir = params.dir_roms_only_ssp245_monthly
        elif scenario == 'ssp585':
            fdir = params.dir_roms_only_ssp585_monthly
    elif model_config == 'romsoc_fully_coupled':
        if scenario == 'present':
            fdir = params.dir_romsoc_fully_coupled_present_monthly
        elif scenario == 'ssp245':
            fdir = params.dir_romsoc_fully_coupled_ssp245_monthly
        elif scenario == 'ssp585':
            fdir = params.dir_romsoc_fully_coupled_ssp585_monthly
    fname = "{}_{}_{}_{}_monthly_avg.nc".format(params.grid_name,model_config,scenario,year)
    fcomb = '{}{}'.format(fdir,fname)
    return fcomb, fdir, fname

def preprocessor_monthly(ds):
    fname = ds.encoding['source']
    allyears = re.findall('[0][01][0-9]', fname)
    month = allyears[-1]
    timerange = pd.date_range(start=f"{year}-01-01",end=f"{year}-12-31")
    #ds = ds.expand_dims(time=[time])
    ds.assign_coords(time=timerange)
    return ds






def get_list_of_combined_fdirs_and_fnames_raw_for_specific_year_atmospheric_data(params,model_config,scenario,year=2010):
    fcombs = []
    if model_config == 'roms_only':
        if scenario == 'present':
            fdir = params.dir_roms_only_present_daily_atmospheric_forcing
            fname = [[fdir+'pactcs30_1day_1979-2019_frc_corr_shf.nc',fdir+'pactcs30_1day_2020-2021_frc_corr_shf.nc'],
                     [fdir+'pactcs30_1day_1979-2019_frc_corr_srf.nc',fdir+'pactcs30_1day_2020-2021_frc_corr_srf.nc'],
                     [fdir+'pactcs30_1day_1979-2019_frc_corr_swf.nc',fdir+'pactcs30_1day_2020-2021_frc_corr_swf.nc'],
                     [fdir+'pactcs30_1day_1979-2019_frc_corr_sms.nc',fdir+'pactcs30_1day_2020-2021_frc_corr_sms.nc'],
                     [fdir+'pactcs30_1day_1979-2019_frc_corr_SST.nc',fdir+'pactcs30_1day_2020-2021_frc_corr_SST.nc']]
        elif scenario == 'ssp245':
            fdir = params.dir_roms_only_ssp245_daily_atmospheric_forcing
            fname = [[fdir+'pactcs30_1day_1979-2021_frc_corr_shf.nc'],
                     [fdir+'pactcs30_1day_1979-2021_frc_corr_srf.nc'],
                     [fdir+'pactcs30_1day_1979-2021_frc_corr_swf.nc'],
                     [fdir+'pactcs30_1day_1979-2021_frc_corr_sms.nc'],
                     [fdir+'pactcs30_1day_1979-2021_frc_corr_SST.nc']]
        elif scenario == 'ssp585':
            fdir = params.dir_roms_only_ssp585_daily_atmospheric_forcing
            fname = [[fdir+'pactcs30_1day_1979-2021_frc_corr_shf.nc'],
                     [fdir+'pactcs30_1day_1979-2021_frc_corr_srf.nc'],
                     [fdir+'pactcs30_1day_1979-2021_frc_corr_swf.nc'],
                     [fdir+'pactcs30_1day_1979-2021_frc_corr_sms.nc'],
                     [fdir+'pactcs30_1day_1979-2021_frc_corr_SST.nc']]
        fcombs = fname #.append('{}{}'.format(fdir,fname))
    elif model_config == 'romsoc_fully_coupled':
        if scenario == 'present':
            fdir = params.dir_romsoc_fully_coupled_present_daily_atmospheric_forcing+str(year)+'/' 
        elif scenario == 'ssp245':
            fdir = params.dir_romsoc_fully_coupled_ssp245_daily_atmospheric_forcing+str(year)+'/' 
        elif scenario == 'ssp585':
            fdir = params.dir_romsoc_fully_coupled_ssp585_daily_atmospheric_forcing+str(year)+'/' 
        sorted_list = sorted(glob.glob(f'{fdir}lffd*.ncAVG'))
        fcombs = sorted_list   #{}avg_{:03d}.nc".format(fdir,i))
    return fcombs

def set_fdir_and_fname_raw_for_monthly_data_atmospheric_data(params,model_config,scenario,year):
    if model_config == 'roms_only':
        if scenario == 'present':
            fdir = params.dir_roms_only_present_monthly_atmospheric_forcing
        elif scenario == 'ssp245':
            fdir = params.dir_roms_only_ssp245_monthly_atmospheric_forcing
        elif scenario == 'ssp585':
            fdir = params.dir_roms_only_ssp585_monthly_atmospheric_forcing
    elif model_config == 'romsoc_fully_coupled':
        if scenario == 'present':
            fdir = params.dir_romsoc_fully_coupled_present_monthly_atmospheric_forcing
        elif scenario == 'ssp245':
            fdir = params.dir_romsoc_fully_coupled_ssp245_monthly_atmospheric_forcing
        elif scenario == 'ssp585':
            fdir = params.dir_romsoc_fully_coupled_ssp585_monthly_atmospheric_forcing
    fname = "{}_{}_{}_{}_forcing_monthly_means.nc".format(params.grid_name,model_config,scenario,year)
    fcomb = '{}{}'.format(fdir,fname)
    return fcomb, fdir, fname
