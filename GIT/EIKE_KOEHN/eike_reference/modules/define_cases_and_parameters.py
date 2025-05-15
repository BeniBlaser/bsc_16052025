"""
this file defines the different cases (reference and sensitivity cases)
date: Apr 26, 2022
author: Eike E Koehn (adapted from Urs Hofmann Elizondo)
"""

from dataclasses import dataclass
import yaml
import argparse

#%%
@dataclass
class Config_parameters():
    """Class to define paths and model parameters used in the model""" 
    
    # case instance name
    config_file: str

    # model grid name
    grid_name: str
    grid_file: str

    # list of model simulation names
    model_configs: list
    scenarios: list
    
    # prespinup directory
    dir_roms_only_pre_spinup: str

    # spinup directories
    dir_roms_only_present_spinup: str
    dir_roms_only_ssp245_spinup: str
    dir_roms_only_ssp585_spinup: str

    # spinup directories on zlevels
    dir_roms_only_present_spinup_zlevs: str
    dir_roms_only_ssp245_spinup_zlevs: str
    dir_roms_only_ssp585_spinup_zlevs: str

    # model raw directory
    dir_roms_only_present_raw: str
    dir_roms_only_ssp245_raw: str
    dir_roms_only_ssp585_raw: str
    dir_romsoc_fully_coupled_present_raw: str
    dir_romsoc_fully_coupled_ssp245_raw: str
    dir_romsoc_fully_coupled_ssp585_raw: str

    # model daily directory
    dir_roms_only_present_daily: str
    dir_roms_only_ssp245_daily: str
    dir_roms_only_ssp585_daily: str
    dir_romsoc_fully_coupled_present_daily: str
    dir_romsoc_fully_coupled_ssp245_daily: str
    dir_romsoc_fully_coupled_ssp585_daily: str

    # model daily z-levels directory
    dir_roms_only_present_daily_zlevs: str
    dir_roms_only_ssp245_daily_zlevs: str
    dir_roms_only_ssp585_daily_zlevs: str
    dir_romsoc_fully_coupled_present_daily_zlevs: str
    dir_romsoc_fully_coupled_ssp245_daily_zlevs: str
    dir_romsoc_fully_coupled_ssp585_daily_zlevs: str

    # model raw directory (monthly)
    dir_roms_only_present_monthly: str
    dir_roms_only_ssp245_monthly: str
    dir_roms_only_ssp585_monthly: str
    dir_romsoc_fully_coupled_present_monthly: str
    dir_romsoc_fully_coupled_ssp245_monthly: str   
    dir_romsoc_fully_coupled_ssp585_monthly: str   

    # model zlevel directory (monthly)
    dir_roms_only_present_monthly_zlevs: str
    dir_roms_only_ssp245_monthly_zlevs: str
    dir_roms_only_ssp585_monthly_zlevs: str
    dir_romsoc_fully_coupled_present_monthly_zlevs: str
    dir_romsoc_fully_coupled_ssp245_monthly_zlevs: str   
    dir_romsoc_fully_coupled_ssp585_monthly_zlevs: str   

    # model atmospheric/forcing fields (daily)
    dir_roms_only_present_daily_atmospheric_forcing: str
    dir_roms_only_ssp245_daily_atmospheric_forcing: str
    dir_roms_only_ssp585_daily_atmospheric_forcing: str
    dir_romsoc_fully_coupled_present_daily_atmospheric_forcing: str
    dir_romsoc_fully_coupled_ssp245_daily_atmospheric_forcing: str
    dir_romsoc_fully_coupled_ssp585_daily_atmospheric_forcing: str

    # model atmospheric/forcing fields (monthly)
    dir_roms_only_present_monthly_atmospheric_forcing: str
    dir_roms_only_ssp245_monthly_atmospheric_forcing: str
    dir_roms_only_ssp585_monthly_atmospheric_forcing: str
    dir_romsoc_fully_coupled_present_monthly_atmospheric_forcing: str
    dir_romsoc_fully_coupled_ssp245_monthly_atmospheric_forcing: str
    dir_romsoc_fully_coupled_ssp585_monthly_atmospheric_forcing: str

    # other directory roots
    dir_root_model_output_regridded:       str
    dir_root_thresholds_and_climatologies: str
    dir_root_boolean_arrays:               str
    dir_root_labeled_arrays:               str
    dir_root_event_coordinates:            str
    dir_root_event_characteristics:        str
    dir_root_mapped_characteristics:       str

    # monthly variables
    monthly_vars: list

    # grid parameters
    grid_zlevs: list
    grid_zlevels_type: str
    grid_nzlevs: int
    grid_variables_to_regrid: str
    grid_fac_horizontal_downsampling: int
    grid_type_horizontal_downsampling: str

    # spinup parameters
    spinup_start_year: int
    spinup_end_year: int

    # hindcast analysis period parameters
    hindcast_analysis_start_year: int
    hindcast_analysis_end_year: int
    hindcast_analysis_period_temporal_resolution: str

    # threshold parameters
    threshold_variable_name: str
    threshold_percentile: float
    threshold_period_start_year: int
    threshold_period_end_year: int
    threshold_baseline: str
    threshold_daysinyear: int
    threshold_aggregation_window_size: int
    threshold_smoothing_window_size: int

    # boolean smoothing parameters
    boolean_smoothing_type: str
    #boolean_temporal_smoothing_kernel_size: int
    #boolean_horizontal_smoothing_kernel_size: int
    #boolean_vertical_smoothing_kernel_size: int

    # index extraction (calculation of characteristics) parameters
    keep_only_surface_events: bool
    keep_only_long_events: bool

    # labeled region limits
    labeled_southern_boundary: int
    labeled_northern_boundary: int
    labeled_western_boundary: int
    labeled_eastern_boundary: int

    # DERIVED FILES

    # define simulation_name
    def _simulation_name(self,model_config,scenario) -> str:
        return "{}_{}".format(model_config,scenario)
    
    # define directories
    def _dir_model_output_regridded(self,model_simulation_name) -> str:
        return "{}{}/".format(self.dir_root_model_output_regridded,model_simulation_name)

    def _dir_thresholds_and_climatologies(self,model_simulation_name) -> str:
        return "{}{}/".format(self.dir_root_thresholds_and_climatologies,model_simulation_name)

    def _dir_boolean_arrays(self,model_simulation_name) -> str:
        return "{}{}/".format(self.dir_root_boolean_arrays,model_simulation_name)

    def _dir_labeled_arrays(self,model_simulation_name) -> str:
        return "{}{}/".format(self.dir_root_labeled_arrays,model_simulation_name)

    def _dir_event_coordinates(self,model_simulation_name) -> str:
        return "{}{}/".format(self.dir_root_event_coordinates,model_simulation_name)

    def _dir_event_characteristics(self,model_simulation_name) -> str:
        return "{}{}/".format(self.dir_root_event_characteristics,model_simulation_name)

    def _dir_mapped_characteristics(self,model_simulation_name) -> str:
        return "{}{}/".format(self.dir_root_mapped_characteristics,model_simulation_name)

    # define filenames
    def _fname_model_output_spinup(self) -> list:
        filelist = []
        for year in range(self.spinup_start_year,self.spinup_end_year+1):
            filelist.append("{}_{}_avg.nc".format(self.grid_name,year))
        return filelist
    
    def _fname_model_output_raw(self) -> list:
        filelist = []
        for year in range(int(self.hindcast_analysis_start_year-1),self.hindcast_analysis_end_year+1):
            filelist.append("{}_{}_avg.nc".format(self.grid_name,year))
        return filelist
    
    def _fname_model_output_romsoc_raw(self) -> list:
        filelist = []
        for year in range(self.hindcast_analysis_start_year,self.hindcast_analysis_end_year+1):
            filelist.append("{}_{}_avg.nc".format(self.grid_name,year))
        return filelist
    
    def _fname_model_output_regridded(self) -> str:
        filename = "z_{}_YYYY_avg_{}zlevs_{}_{}x{}{}_downsampling.nc".format(self.grid_name,self.grid_nzlevs,self.grid_zlevels_type,self.grid_fac_horizontal_downsampling,self.grid_fac_horizontal_downsampling,self.grid_type_horizontal_downsampling)
        #for year in range(self.hindcast_analysis_start_year,self.hindcast_analysis_end_year+1):
        #    filelist.append("z_{}_YYYY_avg_{}zlevs_{}_{}x{}{}_downsampling.nc".format(self.grid_name,self.grid_nzlevs,self.grid_zlevels_type,self.grid_fac_horizontal_downsampling,self.grid_fac_horizontal_downsampling,self.grid_type_horizontal_downsampling))
        return filename
    
    def _fname_model_output_regridded_romsoc(self) -> str:
        filename = "z_avg_YYYY_MMM_{}zlevs_{}_{}x{}{}_downsampling.nc".format(self.grid_nzlevs,self.grid_zlevels_type,self.grid_fac_horizontal_downsampling,self.grid_fac_horizontal_downsampling,self.grid_type_horizontal_downsampling)
        filelist = []
        #for year in range(self.hindcast_analysis_start_year,self.hindcast_analysis_end_year+1):
        #    filelist.append("z_{}_YYYY_MMM_avg_{}zlevs_{}_{}x{}{}_downsampling.nc".format(self.grid_name,self.grid_nzlevs,self.grid_zlevels_type,self.grid_fac_horizontal_downsampling,self.grid_fac_horizontal_downsampling,self.grid_type_horizontal_downsampling))
        return filename

    def _fname_threshold_and_climatology(self,varia) -> str:
        return "hobday2016_threshold_and_climatology_{}_{}zlevs_{}_{}x{}{}_downsampling_{}-{}analysisperiod_{}perc_{}-{}baseperiod_{}baseline_{}aggregation_{}smoothing.nc".format(varia,self.grid_nzlevs,self.grid_zlevels_type,self.grid_fac_horizontal_downsampling,self.grid_fac_horizontal_downsampling,self.grid_type_horizontal_downsampling,self.hindcast_analysis_start_year,self.hindcast_analysis_end_year,self.threshold_percentile,self.threshold_period_start_year,self.threshold_period_end_year,self.threshold_baseline,self.threshold_aggregation_window_size,self.threshold_smoothing_window_size)
    
    def _fname_boolean_array(self,varia) -> str:
        return "boolean_array_{}.nc".format(self._fname_threshold_and_climatology(varia)[:-3])#,self.boolean_temporal_smoothing_kernel_size,self.boolean_horizontal_smoothing_kernel_size,self.boolean_vertical_smoothing_kernel_size)

    def _fname_labeled_array(self,varia) -> str:
        return "labeled_array_{}_boolsmooth_{}.nc".format(self._fname_threshold_and_climatology(varia)[:-3],self.boolean_smoothing_type)#,self.boolean_temporal_smoothing_kernel_size,self.boolean_horizontal_smoothing_kernel_size,self.boolean_vertical_smoothing_kernel_size)

    def _fname_extracted_coordinates(self,varia) -> str:
        return "event_coordinates_from_{}.pck".format(self._fname_labeled_array(varia)[:-3])

    def _fname_event_characteristics(self,varia) -> str:
        return "event_characteristics_from_{}.pck".format(self._fname_labeled_array(varia)[:-3])

    def _fname_mapped_characteristics(self,varia) -> str:
        return "mapped_characteristics_from_{}.pck".format(self._fname_labeled_array(varia)[:-3])
    
def read_config_files(config_file,config_class=Config_parameters):
    """This function reads a configuration file, and fills in the attributes of the dataclass with the respective entries in the configuratio file
    
    Keyword arguments:
    config_file -- path to configuration file. Should be a yaml file
    config_class -- dataclass (default: Config_parameters dataclass defined above)
    """ 
    assert '.yaml' in config_file.lower(), "The configuration file should be a '.yaml' file"
    
    with open(config_file) as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)

    case_instance = config_class(**config_list)
    
    return case_instance

#%%

# def parse_inputs():
#     """This reads user input from the terminal to start running the shelled pteropod IBM
    
#     """ 

#     parser = argparse.ArgumentParser(description="Run shelled pteropod IBM")
#     parser.add_argument("--year", required=True, type=int,
#                         help="Year of the simulation. Year should coincide with name of file.")
#     parser.add_argument("--version", required=True, type=int,
#                         help="Version of the run. This integer is used to set the random seed.")
#     parser.add_argument("--control", required=False, nargs='?',
#                         const=0, default=0, type=int,
#                         help="Determine which scenario is used (0: with extremes; 1: without extremes).")
#     parser.add_argument("--restart_day", required=False, nargs='?', const=1, default=1, type=int,
#                         help="Determine the restart day for the simulation.")
#     parser.add_argument("--config_file", required=False, nargs='?',
#                         const="IBM_config_parameters.yaml",
#                         default="IBM_config_parameters.yaml", type=str,
#                         help="Yaml file containing the paths and parameters needed for the IBM.")

#     args = parser.parse_args()

#     restart_day = args.restart_day
#     main_flag = True if restart_day <= 1 else False
#     restart_day = 1 if restart_day < 1 else restart_day

#     return args.year, args.version, args.control, restart_day, main_flag, args.config_file