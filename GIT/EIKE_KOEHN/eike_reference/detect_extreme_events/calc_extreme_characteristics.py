"""
Content: calculate the extreme characteristics based on the extracted evnt indices
Author: Eike E Koehn
Date: Apr 26, 2022
"""

#%% DEFINE THE SCRIPTNAME
scriptdir = '/home/koehne/Documents/publications/paper2/scripts_clean/detect_extreme_events/'   # /home/koehne/... # Euler using python/3.7.1 (gcc/6.3.0)
scriptname = 'calc_extreme_characteristics.py'

#%% LOAD THE PACKAGES
import xarray as xr
import numpy as np
from datetime import date
import pickle
import itertools
from scipy.optimize import curve_fit
import pandas as pd
import re

#%% LOAD THE CASES CLASS
exec(open('../modules/define_cases_and_parameters.py').read())
exec(open('../modules/case_functions.py').read())

#%% DEFINE FUNCTIONS

def preprocessor(ds):
    fname = ds.encoding['source']
    allyears = re.findall('[12][78901][0-9][0-9]', fname)
    year = allyears[-1]
    timerange = pd.date_range(start=f"{year}-01-01",end=f"{year}-12-31")
    #ds = ds.expand_dims(time=[time])
    ds.assign_coords(time=timerange)
    return ds

def load_mld(params):
    print('Load the MLD.')
    dir_regridded_model_output = params.dir_model_output_regridded
    filelist = []
    for filename in params._fname_model_output_regridded():
        filelist.append(dir_regridded_model_output+filename)
    mfn = xr.open_mfdataset(filelist,data_vars=['mld_holte'],preprocess=preprocessor,concat_dim='time')#, combine='nested')#,concat_dim='time')
    mld = mfn.mld_holte.values
    return mld

# Fitting functions (linear and quadratic regression)
def func_linear(x, b, c):
    return b*x + c 

def func_square(x, a, b, c):
    return a*x**2 + b*x + c 

def calc_regressions(extreme_indices_dict,z):
    indices = np.stack((extreme_indices_dict['itime'],extreme_indices_dict['idepth']))
    depvec = []
    timvec = np.unique(indices[0,:])
    timvec0 = timvec[0]
    for idx,i in enumerate(timvec):
        cho = (indices[0,:]==i)
        depvec.append(np.min(indices[1,cho]))
    depvec = z[depvec]
    timvec = timvec-timvec0
    if np.size(timvec)==1:
        popt_linear = np.array([0.])
        popt_square = np.array([0.])
    elif np.size(timvec)==2:
        popt_linear, _ = curve_fit(func_linear, timvec, depvec)
        popt_square = np.array([0.])        
    else:
        popt_linear, _ = curve_fit(func_linear, timvec, depvec)
        popt_square, _ = curve_fit(func_square, timvec, depvec)
    linear_terms = np.array(popt_linear[0])
    quadratic_terms = np.array(popt_square[0])
    return linear_terms,quadratic_terms

# Calculate characteristics
def calculate_depth_anomaly_rel_to_ml(extreme_indices_dict,z):
    #print('Calculate depth anomaly relative to MLD.')
    # extract the correct event coordinates
    etaidx = extreme_indices_dict['ieta'][0]
    xiidx = extreme_indices_dict['ixi'][0]
    timidx = extreme_indices_dict['itime']
    unique_times = np.unique(timidx)
    depidx = extreme_indices_dict['idepth']
    unique_depths = np.zeros_like(unique_times)+np.NaN
    for tidx,ut in enumerate(unique_times):
        unique_depths[tidx] = np.max(depidx[timidx==ut])
    # extract the depth time series and corresponding MLD time series
    unique_depths = np.array(unique_depths,dtype='int')
    all_depths = z[unique_depths]               # ATTENTION: z has to be negatively defined
    all_mld = mld[unique_times,etaidx,xiidx]    # ATTENTION: mld has to be negatively defined
    # calculate the depth anomaly vector
    depth_anom_vec = all_depths-all_mld
    # get the statistics for the anomaly vector
    deep_anomaly_rel_to_mld_max = np.max(depth_anom_vec)
    deep_anomaly_rel_to_mld_min = np.min(depth_anom_vec)
    deep_anomaly_rel_to_mld_mean = np.mean(depth_anom_vec)
    deep_anomaly_rel_to_mld_median = np.median(depth_anom_vec)
    return deep_anomaly_rel_to_mld_max, deep_anomaly_rel_to_mld_min, deep_anomaly_rel_to_mld_mean, deep_anomaly_rel_to_mld_median

def calc_max_consec_ones(array):
    #print('Calculate the number of maximum consecutive surface values (CORE).')
    # get the number of consequent occurrences of numbers
    zt_list = [(x[0], len(list(x[1]))) for x in itertools.groupby(array)]
    # remove all that are not "1"
    for izt in zt_list:
        if izt[0]!=1:
            zt_list.remove(izt)
    # find the maximum number of 
    if len(zt_list)>0:
        max_consec = max(zt_list, key=lambda x:x[1])[1]
    else:
        max_consec = 1
    return max_consec

def get_chars_core(exdict,key,mld,z,dz,lon,lat,area):
    print('Calculate the extreme characteristics (CORE).')
    duration_c = np.zeros(len(exdict.keys()))
    duration_s = np.zeros_like(duration_c)
    starttime_c = np.zeros_like(duration_c)
    starttime_s = np.zeros_like(duration_c)
    endtime_c = np.zeros_like(duration_c)
    endtime_s = np.zeros_like(duration_c)
    etas = np.zeros_like(duration_c)
    xis = np.zeros_like(duration_c)
    areavec = np.zeros_like(duration_c)
    lons = np.zeros_like(duration_c)
    lats = np.zeros_like(duration_c)
    times = np.zeros_like(duration_c)
    collected_events = np.zeros_like(duration_c)
    maxgap = np.zeros_like(duration_c)
    maxconsecsurf = np.zeros_like(duration_c)
    surf10time = np.zeros_like(duration_c)
    surf90time = np.zeros_like(duration_c)
    maxdepth_abs = np.zeros_like(duration_c)
    meandepth_abs = np.zeros_like(duration_c)
    mediandepth_abs = np.zeros_like(duration_c)
    fraction_in_ml = []
    depth_timeseries = []
    max_fraction_in_ml = np.zeros_like(duration_c)
    mean_fraction_in_ml = np.zeros_like(duration_c)
    median_fraction_in_ml = np.zeros_like(duration_c)
    upper_boundary_linear_reg = np.zeros_like(duration_c)
    upper_boundary_squared_reg = np.zeros_like(duration_c)
    deep_anomaly_rel_to_mld_max = np.zeros_like(duration_c)
    deep_anomaly_rel_to_mld_min = np.zeros_like(duration_c)
    deep_anomaly_rel_to_mld_mean = np.zeros_like(duration_c)
    deep_anomaly_rel_to_mld_median = np.zeros_like(duration_c)
    for eidx,ev in enumerate(exdict.keys()):
        if np.mod(eidx,10000)==0:
            print(eidx,ev)
        duration_c[eidx]=np.max(exdict[ev]['itime'])-np.min(exdict[ev]['itime'])+1
        ieta = exdict[ev]['ieta'][0]
        ixi = exdict[ev]['ixi'][0]
        etas[eidx] = ieta
        xis[eidx] = ixi
        lons[eidx] = lon[ieta,ixi]
        lats[eidx] = lat[ieta,ixi]
        areavec[eidx] = area[ieta,ixi]
        times[eidx] = exdict[ev]['itime'][0]/365+1979.
        duration_s[eidx]=np.sum(exdict[ev]['idepth']==0)
        diffs = np.diff(np.sort(exdict[ev]['itime'][exdict[ev]['idepth']==0]))
        collected_events[eidx] = np.sum(diffs>1)+1
        starttime_c[eidx] = np.min(exdict[ev]['itime'])
        starttime_s[eidx] = np.min(exdict[ev]['itime'][exdict[ev]['idepth']==0])
        endtime_c[eidx] = np.max(exdict[ev]['itime'])
        endtime_s[eidx] = np.max(exdict[ev]['itime'][exdict[ev]['idepth']==0]) 
        surf10time[eidx] = np.percentile(exdict[ev]['itime'][exdict[ev]['idepth']==0],10)
        surf90time[eidx] = np.percentile(exdict[ev]['itime'][exdict[ev]['idepth']==0],90)
        maxdepth_idx = np.max(exdict[ev]['idepth'])
        maxdepth_abs[eidx] = z[maxdepth_idx]
        if len(diffs)>0:
            maxgap[eidx]=np.max(diffs)-1
            diffs[diffs!=1]=0
            maxconsecsurf[eidx] = calc_max_consec_ones(diffs) 
        else:
            maxgap[eidx]=0
            maxconsecsurf[eidx] = 1
        # calculate the fraction of the extreme in the ML
        if key == 'L1D':
            in_ml = np.zeros_like(exdict[ev]['itime'])
            dzs = np.zeros_like(exdict[ev]['itime'])
            for i in range(len(exdict[ev]['itime'])):
                mld_value = (1*mld[exdict[ev]['itime'][i],exdict[ev]['ieta'][i],exdict[ev]['ixi'][i]])
                in_ml[i] = 1*(z[exdict[ev]['idepth'][i]] > mld_value) 
                dzs[i] = dz[exdict[ev]['idepth'][i]]
            fraction_in_ml_for_this_event = []
            timeseries_of_max_depths = []
            for t in np.unique(exdict[ev]['itime']):
                indices_to_consider = np.where(exdict[ev]['itime']==t)[0]
                #trues = np.sum(in_ml[indices_to_consider]) # sum together all ones
                true_meters = np.sum(in_ml[indices_to_consider]*dzs[indices_to_consider]) # sum together all dzs in ml
                all_meters = np.sum(dzs[indices_to_consider])    # sum together all dzs (inside or outside ml)
                #fraction_in_ml_for_this_event.append(trues/len(indices_to_consider))
                fraction_in_ml_for_this_event.append(true_meters/all_meters)
                # calculate the mean depth, i.e. the mean over the time series of max. depths for each event
                timeseries_of_max_depths.append(z[int(np.max(np.array(exdict[ev]['idepth'])[indices_to_consider]))])
        else:
            fraction_in_ml_for_this_event = [1.]
            timeseries_of_max_depths = [0.]
        #print(fraction_in_ml_for_this_event)
        max_fraction_in_ml[eidx] = np.max(fraction_in_ml_for_this_event)
        mean_fraction_in_ml[eidx] = np.mean(fraction_in_ml_for_this_event)
        median_fraction_in_ml[eidx] = np.median(fraction_in_ml_for_this_event)
        fraction_in_ml.append(fraction_in_ml_for_this_event)
        meandepth_abs[eidx] = np.mean(timeseries_of_max_depths)
        mediandepth_abs[eidx] = np.median(timeseries_of_max_depths)
        depth_timeseries.append(timeseries_of_max_depths)
        if key == 'L1D':
            upper_boundary_linear_reg[eidx],upper_boundary_squared_reg[eidx] = calc_regressions(exdict[ev],z)#dicts,evid)
            deep_anomaly_rel_to_mld_max[eidx], deep_anomaly_rel_to_mld_min[eidx], deep_anomaly_rel_to_mld_mean[eidx], deep_anomaly_rel_to_mld_median[eidx] = calculate_depth_anomaly_rel_to_ml(exdict[ev],z)
        else:
            upper_boundary_linear_reg[eidx],upper_boundary_squared_reg[eidx] = 0.,0.
            deep_anomaly_rel_to_mld_max[eidx], deep_anomaly_rel_to_mld_min[eidx], deep_anomaly_rel_to_mld_mean[eidx], deep_anomaly_rel_to_mld_median[eidx] = 0.,0.,0.,0.
    chars = dict()
    chars['duration'] = dict()
    chars['duration']['column']=duration_c
    chars['duration']['surface']=duration_s
    chars['starttime'] = dict()
    chars['starttime']['column']=starttime_c
    chars['starttime']['surface']=starttime_s
    chars['endtime'] = dict()
    chars['endtime']['column']=endtime_c
    chars['endtime']['surface']=endtime_s
    chars['ietas']=etas
    chars['ixis']=xis
    chars['lons']=lons
    chars['lats']=lats
    chars['times']=times    
    chars['areas']=areavec
    chars['collevents_surf']=collected_events
    chars['maxgap_surf']=maxgap
    chars['maxconsec_surf']=maxconsecsurf
    chars['surf10time']=surf10time
    chars['surf90time']=surf90time
    chars['maxdepth_abs']=maxdepth_abs
    chars['meandepth_abs']=meandepth_abs
    chars['mediandepth_abs']=mediandepth_abs
    chars['fraction_in_ml']=fraction_in_ml
    chars['max_fraction_in_ml']=max_fraction_in_ml
    chars['mean_fraction_in_ml']=mean_fraction_in_ml
    chars['median_fraction_in_ml']=median_fraction_in_ml
    chars['upper_boundary_linear_reg']=upper_boundary_linear_reg
    chars['upper_boundary_squared_reg']=upper_boundary_squared_reg
    chars['deep_anomaly_rel_to_mld_max'] = deep_anomaly_rel_to_mld_max      # shallowest event depth relative to MLD
    chars['deep_anomaly_rel_to_mld_min'] = deep_anomaly_rel_to_mld_min      # deepest event depth relative to MLD
    chars['deep_anomaly_rel_to_mld_mean'] = deep_anomaly_rel_to_mld_mean
    chars['deep_anomaly_rel_to_mld_median'] = deep_anomaly_rel_to_mld_median
    return chars

def keep_only_surface_events(exdict):
    print('Keep only events with surface expression.')
    exdict_surface = dict()
    for key in exdict:
        if key=='E0D' or key=='L1D':
            print(key)
            exdict_surface[key] = dict()
            exnum = -1
            list_of_indices = exdict[key]#[key]
            for ev_indices in list_of_indices:
                #if np.mod(exnum,10000)==0:
                #    print(exnum)
                #print(evidx)
                if np.any(ev_indices[1]==0):     # ATTENTION: Here, I assume that the surface is located at the depth index 0...
                    exnum += 1
                    exdict_surface[key][str(exnum)]=dict()
                    exdict_surface[key][str(exnum)]['itime']=ev_indices[0]
                    exdict_surface[key][str(exnum)]['idepth']=ev_indices[1]
                    exdict_surface[key][str(exnum)]['ieta']=ev_indices[2]
                    exdict_surface[key][str(exnum)]['ixi']=ev_indices[3]  
        else:
            exdict_surface[key] = exdict[key]
    return exdict_surface

def keep_only_long_events(exdict_surface,minduration=5):
    print('Keep only long events.')
    exdict_surface_long = dict()
    for key in exdict_surface:
        exdict_surface_long[key] = dict()
        exnum = -1
        print(key)
        for evidx,ev in enumerate(exdict_surface[key]):
            print(ev)
            #if np.mod(exnum,10000)==0:
            #    print(exnum)
            #print(evidx)
            if np.size(np.unique(exdict_surface[key][ev]['itime']))>=minduration:
                exnum += 1
                exdict_surface_long[key][str(exnum)]=dict()
                exdict_surface_long[key][str(exnum)]['itime']=exdict_surface[key][ev]['itime']
                exdict_surface_long[key][str(exnum)]['idepth']=exdict_surface[key][ev]['idepth']
                exdict_surface_long[key][str(exnum)]['ieta']=exdict_surface[key][ev]['ieta']
                exdict_surface_long[key][str(exnum)]['ixi']=exdict_surface[key][ev]['ixi']   
    return exdict_surface_long    

def get_characteristics(exdict,mld,z,dz,lon,lat,area):
    print('Calculate the extreme characteristics.')
    chardict = dict()
    for key in exdict:
        print(key)
        if key == 'E0D':
            print("E0D")
            chardict[key] = get_chars_core(exdict[key],key,mld,z,dz,lon,lat,area)
        elif key == 'L1D':
            print('L1D')
            chardict[key] = get_chars_core(exdict[key],key,mld,z,dz,lon,lat,area)
    return chardict

def get_intensities(exdict,params):
    print('Calculate the extreme intensities.')
    cat_index = load_cat_index_array(params)
    intensity_abs = load_intensity_abs_array(params)
    print('Start the calculation.')
    intensdict = dict()
    for key in ['E0D','L1D']:
        intensdict[key] = dict()
        print(key)
        # intensities in terms of CII
        intensdict[key]['intensities'] = dict()
        intensdict[key]['max_intens'] = dict()
        intensdict[key]['mean_intens'] = dict()
        # intensities in absolute temrs (T-T_thresh)
        intensdict[key]['intensities_abs'] = dict()
        intensdict[key]['max_intens_abs'] = dict()
        intensdict[key]['mean_intens_abs'] = dict()        
        for eidx,evnum in enumerate(exdict[key].keys()):
            if np.mod(eidx,10000)==0:
                print('Intensity for {} event number: {}'.format(key,eidx))
            t = exdict[key][evnum]['itime']
            s = exdict[key][evnum]['idepth']
            e = exdict[key][evnum]['ieta']
            x = exdict[key][evnum]['ixi']
            cii = cat_index[t,s,e,x]
            intensdict[key]['intensities'][evnum] = cii
            intensdict[key]['max_intens'][evnum] = np.max(cii)
            intensdict[key]['mean_intens'][evnum] = np.mean(cii)
            i_abs = intensity_abs[t,s,e,x]
            intensdict[key]['intensities_abs'][evnum] = i_abs
            intensdict[key]['max_intens_abs'][evnum] = np.max(i_abs)
            intensdict[key]['mean_intens_abs'][evnum] = np.mean(i_abs)
    return intensdict

def load_extreme_indices(params):
    print('Load extreme coordinates.')
    indices_dir = params.dir_event_coordinates
    indices_filename = params._fname_extracted_coordinates()
    exdict = pickle.load(open(indices_dir+indices_filename,"rb"))
    print(exdict.keys())
    return exdict

def load_cat_index_array(params):
    print('Load CAT index array.')
    cat_index_dir = params.dir_boolean_arrays
    cat_index_filename = params._fname_boolean_array()
    cat_index_fn = xr.open_dataset(cat_index_dir+cat_index_filename)
    cat_index = cat_index_fn.cat_index_smoothed.values
    return cat_index

def load_intensity_abs_array(params):
    print('Load intensity abs array.')
    intensity_abs_dir = params.dir_boolean_arrays
    intensity_abs_filename = params._fname_boolean_array()
    intensity_abs_fn = xr.open_dataset(intensity_abs_dir+intensity_abs_filename)
    intensity_abs = intensity_abs_fn.intensity_abs.values
    return intensity_abs

def save_all_characteristics(params,exdict,chardict,intensdict):
    print('Save all characteristics.')
    # Save the different dictionaries
    alldict = dict()
    alldict['exdict'] = exdict
    alldict['chardict'] = chardict
    alldict['intensdict'] = intensdict
    alldict['author'] = 'E. E. Koehn'
    alldict['date'] = str(date.today())
    alldict['scriptdir'] = scriptdir
    alldict['scriptname'] = scriptname
    alldict['casename'] = casename
    alldict['case_description'] = 'Given by the following attributes:'
    for pattribute in dir(params):
        if not pattribute.startswith('__') and not callable(getattr(params,pattribute)):
            print(pattribute)
            print(eval('params.'+pattribute))
            alldict['params.'+pattribute] = eval('params.'+pattribute)
        if not pattribute.startswith('__') and callable(getattr(params,pattribute)):
            print(pattribute)
            print(eval('params.'+pattribute+'()'))
            alldict['params.'+pattribute+'()'] = eval('params.'+pattribute+'()')
    # Saving, by pickling the 'data' dictionary using the highest protocol available.
    with open(params.dir_event_characteristics+params._fname_event_characteristics(), 'wb') as f:
        pickle.dump(alldict, f, pickle.HIGHEST_PROTOCOL)

#def calc_extreme_characeristics(params):
#    z, dz, lon, lat, area = get_z_dz_lon_lat_and_area(params)
#    exdict = load_extreme_indices(params)
#    if params.keep_only_surface_events == True:
#        exdict = keep_only_surface_events(exdict)
#        exdict = keep_only_long_events(exdict)
#    chardict = get_characteristics(exdict,mld,z,dz,lon,lat,area)
#    intensdict = get_intensities(exdict,params)
#    save_all_characteristics(params,exdict,chardict,intensdict)

#%%
casedir = '/cluster/home/koehne/uphome/Documents/publications/paper2/scripts_clean/modules/cases/'   # /home/koehne/...
casenames = ['case00_reference.yaml']#,'caseA1.yaml','caseB2.yaml','caseC1.yaml']

for casename in casenames:
    params = read_config_files(casedir+casename)
    mld = load_mld(params)
    z, dz, lon, lat, area = get_z_dz_lon_lat_and_area(params)
    exdict = load_extreme_indices(params)
    if params.keep_only_surface_events == True:
        exdict = keep_only_surface_events(exdict)
    if params.keep_only_long_events == True:    # (only becomes relevant when i do not do any smoothing of the boolean array, but i want to discard the short events)
        exdict = keep_only_long_events(exdict)
    chardict = get_characteristics(exdict,mld,z,dz,lon,lat,area)
    intensdict = get_intensities(exdict,params)
    save_all_characteristics(params,exdict,chardict,intensdict)

#%%























































































#%%
#########################################################################################
#########################################################################################
################################################################################################




# #%% DEFINE FUNCTIONS
# def load_extreme_indices(savepath,indices_file):
#     exdict = pickle.load(open(savepath+indices_file,"rb"))
#     return exdict

# def func_linear(x, b, c):
#     return b*x + c 

# def func_square(x, a, b, c):
#     return a*x**2 + b*x + c 

# # get linear and quadratic terms (slopes) of all events
# def calc_regressions(extreme_indices_dict):
# #    linear_terms = []
# #    quadratic_terms = []
# #    for evid in dicts['exdict']['L1D'].keys():
# #        if np.mod(int(evid),10000)==0:
# #            print(evid)
#     #evid = str(evid)
#     #indices = np.stack((dicts['exdict']['L1D'][evid]['itime'],dicts['exdict']['L1D'][evid]['idepth']))
#     indices = np.stack((extreme_indices_dict['itime'],extreme_indices_dict['idepth']))
#     depvec = []
#     timvec = np.unique(indices[0,:])
#     timvec0 = timvec[0]
#     #print(timvec)
#     for idx,i in enumerate(timvec):
#         cho = (indices[0,:]==i)
#         depvec.append(np.min(indices[1,cho]))
#     depvec = z[depvec]
#     timvec = timvec-timvec0
#     popt_linear, pcov_linear = curve_fit(func_linear, timvec, depvec)
#     popt_square, pcov_square = curve_fit(func_square, timvec, depvec)
# #    linear_terms.append(popt_linear[0])
# #    quadratic_terms.append(popt_square[0])
#     linear_terms = np.array(popt_linear[0])
#     quadratic_terms = np.array(popt_square[0])
#     return linear_terms,quadratic_terms

# def load_mld(mld_type):
#     if mld_type == 'raw':
#         list_of_hbls_files = sorted(glob.glob('/nfs/kryo/work/koehne/roms/output/humpac15/hindcast_1979_2019/hindcast_r105_humpac15/daily/avg/humpac15_*_avg.nc'))
#         # loop through annual files and create boolean files
#         for midx,model_file in enumerate(list_of_hbls_files):
#             print(model_file)
#             mfn = xr.open_dataset(model_file)
#             hbls = mfn.variables['hbls'].values
#             if midx == 0:
#                 hbls_all = hbls[:]
#             else:
#                 hbls_all = np.concatenate((hbls_all,hbls),axis=0)
#     elif mld_type == 'filtered':
#         freq = '1over10'
#         hbls_all = xr.open_dataset('/nfs/kryo/work/koehne/roms/analysis/humpac15/hindcast_1979_2019/hindcast_r105_humpac15/processed_model_output/mld/mld_lowpass_fac{}_freq_{}.nc'.format(fac,freq)).variables['mld'].values
#     elif mld_type == 'holte':
#         hbls_all = xr.open_dataset('/nfs/kryo/work/koehne/roms/analysis/humpac15/hindcast_1979_2019/hindcast_r105_humpac15/processed_model_output/mld/humpac15_1979-2019_mld_holte_downsampled_fac3.nc').variables['mld_holte_thres'].values
#     elif mld_type == 'holte_filtered_1over20':
#         hbls_all = xr.open_dataset('/nfs/kryo/work/koehne/roms/analysis/humpac15/hindcast_1979_2019/hindcast_r105_humpac15/processed_model_output/mld/humpac15_1979-2019_mld_holte_downsampled_fac{}_freq_1over20.nc'.format(fac)).variables['mld_holte_thres'].values
#     else:
#         print('Error. Nothing do be done. Returning 0.')
#         hbls_all = 0
#     return hbls_all


# def calculate_depth_anomaly_rel_to_ml(extreme_indices_dict):
#     etaidx = extreme_indices_dict['ieta'][0]
#     xiidx = extreme_indices_dict['ixi'][0]
#     timidx = extreme_indices_dict['itime']
#     unique_times = np.unique(timidx)
#     depidx = extreme_indices_dict['idepth']
#     unique_depths = np.zeros_like(unique_times)+np.NaN
#     for tidx,ut in enumerate(unique_times):
#         unique_depths[tidx] = np.max(depidx[timidx==ut])

#     unique_depths = np.array(unique_depths,dtype='int')
#     all_mld = -1*mld[unique_times,etaidx,xiidx]
#     all_depths = z[unique_depths]
#     depth_anom_vec = all_depths-all_mld

#     deep_anomaly_rel_to_mld_min = np.min(depth_anom_vec)
#     deep_anomaly_rel_to_mld_mean = np.mean(depth_anom_vec)
#     deep_anomaly_rel_to_mld_median = np.median(depth_anom_vec)

#     return deep_anomaly_rel_to_mld_min, deep_anomaly_rel_to_mld_mean, deep_anomaly_rel_to_mld_median


# def get_grid_box_volume_and_area(fn,dz):
#     #dz = np.concatenate((np.array([5]),np.abs(np.diff(fn.depth.values))))
#     dummyfile = fn.attrs['model_file_list'][0]
#     pm = xr.open_dataset(dummyfile).variables['pm'].values
#     pn = xr.open_dataset(dummyfile).variables['pn'].values
#     area = (1/pm * 1/pn)
#     gridboxvolume = dz[:,np.newaxis,np.newaxis]*area[np.newaxis,:,:]
#     return gridboxvolume,area

# def calc_max_consec_ones(array):
#     # get the number of consequent occurrences of numbers
#     z = [(x[0], len(list(x[1]))) for x in itertools.groupby(array)]
#     # remove all that are not "1"
#     for iz in z:
#         if iz[0]!=1:
#             z.remove(iz)
#     # find the maximum number of 
#     if len(z)>0:
#         max_consec = max(z, key=lambda x:x[1])[1]
#     else:
#         max_consec = 1
#     return max_consec

# def get_chars_core(exdict,mld):
#     duration_c = np.zeros(len(exdict.keys()))
#     duration_s = np.zeros_like(duration_c)
#     starttime_c = np.zeros_like(duration_c)
#     starttime_s = np.zeros_like(duration_c)
#     endtime_c = np.zeros_like(duration_c)
#     endtime_s = np.zeros_like(duration_c)
#     etas = np.zeros_like(duration_c)
#     xis = np.zeros_like(duration_c)
#     areavec = np.zeros_like(duration_c)
#     #lons = np.zeros_like(duration)
#     #lats = np.zeros_like(duration)
#     times = np.zeros_like(duration_c)
#     collected_events = np.zeros_like(duration_c)
#     maxgap = np.zeros_like(duration_c)
#     maxconsecsurf = np.zeros_like(duration_c)
#     surf10time = np.zeros_like(duration_c)
#     surf90time = np.zeros_like(duration_c)
#     maxdepth_abs = np.zeros_like(duration_c)
#     meandepth_abs = np.zeros_like(duration_c)
#     mediandepth_abs = np.zeros_like(duration_c)
#     fraction_in_ml = []
#     depth_timeseries = []
#     max_fraction_in_ml = np.zeros_like(duration_c)
#     mean_fraction_in_ml = np.zeros_like(duration_c)
#     median_fraction_in_ml = np.zeros_like(duration_c)
#     upper_boundary_linear_reg = np.zeros_like(duration_c)
#     upper_boundary_squared_reg = np.zeros_like(duration_c)
#     deep_anomaly_rel_to_mld_min = np.zeros_like(duration_c)
#     deep_anomaly_rel_to_mld_mean = np.zeros_like(duration_c)
#     deep_anomaly_rel_to_mld_median = np.zeros_like(duration_c)
#     for eidx,ev in enumerate(exdict.keys()):
#         if np.mod(eidx,10000)==0:
#             print(eidx,ev)
#         duration_c[eidx]=np.max(exdict[ev]['itime'])-np.min(exdict[ev]['itime'])+1
#         ieta = exdict[ev]['ieta'][0]
#         ixi = exdict[ev]['ixi'][0]
#         etas[eidx] = ieta
#         xis[eidx] = ixi
#         areavec[eidx] = area[ieta,ixi]
#         times[eidx] = exdict[ev]['itime'][0]/365+1979.
#         duration_s[eidx]=np.sum(exdict[ev]['idepth']==0)
#         diffs = np.diff(np.sort(exdict[ev]['itime'][exdict[ev]['idepth']==0]))
#         collected_events[eidx] = np.sum(diffs>1)+1
#         starttime_c[eidx] = np.min(exdict[ev]['itime'])
#         starttime_s[eidx] = np.min(exdict[ev]['itime'][exdict[ev]['idepth']==0])
#         endtime_c[eidx] = np.max(exdict[ev]['itime'])
#         endtime_s[eidx] = np.max(exdict[ev]['itime'][exdict[ev]['idepth']==0]) 
#         surf10time[eidx] = np.percentile(exdict[ev]['itime'][exdict[ev]['idepth']==0],10)
#         surf90time[eidx] = np.percentile(exdict[ev]['itime'][exdict[ev]['idepth']==0],90)
#         maxdepth_idx = np.max(exdict[ev]['idepth'])
#         maxdepth_abs[eidx] = z[maxdepth_idx]
#         if len(diffs)>0:
#             maxgap[eidx]=np.max(diffs)-1
#             diffs[diffs!=1]=0
#             maxconsecsurf[eidx] = calc_max_consec_ones(diffs) 
#         else:
#             maxgap[eidx]=0
#             maxconsecsurf[eidx] = 1
#         # calculate the fraction of the extreme in the ML
#         in_ml = np.zeros_like(exdict[ev]['itime'])
#         dzs = np.zeros_like(exdict[ev]['itime'])
#         for i in range(len(exdict[ev]['itime'])):
#             mld_value = (-1*mld[exdict[ev]['itime'][i],exdict[ev]['ieta'][i],exdict[ev]['ixi'][i]])
#             in_ml[i] = 1*(z[exdict[ev]['idepth'][i]] > mld_value) 
#             dzs[i] = dz[exdict[ev]['idepth'][i]]
#         fraction_in_ml_for_this_event = []
#         timeseries_of_max_depths = []
#         for t in np.unique(exdict[ev]['itime']):
#             indices_to_consider = np.where(exdict[ev]['itime']==t)[0]
#             #trues = np.sum(in_ml[indices_to_consider]) # sum together all ones
#             true_meters = np.sum(in_ml[indices_to_consider]*dzs[indices_to_consider]) # sum together all dzs in ml
#             all_meters = np.sum(dzs[indices_to_consider])    # sum together all dzs (inside or outside ml)
#             #fraction_in_ml_for_this_event.append(trues/len(indices_to_consider))
#             fraction_in_ml_for_this_event.append(true_meters/all_meters)
#             # calculate the mean depth, i.e. the mean over the time series of max. depths for each event
#             timeseries_of_max_depths.append(z[int(np.max(np.array(exdict[ev]['idepth'])[indices_to_consider]))])
#         #print(fraction_in_ml_for_this_event)
#         max_fraction_in_ml[eidx] = np.max(fraction_in_ml_for_this_event)
#         mean_fraction_in_ml[eidx] = np.mean(fraction_in_ml_for_this_event)
#         median_fraction_in_ml[eidx] = np.median(fraction_in_ml_for_this_event)
#         fraction_in_ml.append(fraction_in_ml_for_this_event)
#         meandepth_abs[eidx] = np.mean(timeseries_of_max_depths)
#         mediandepth_abs[eidx] = np.median(timeseries_of_max_depths)
#         depth_timeseries.append(timeseries_of_max_depths)
#         upper_boundary_linear_reg[eidx],upper_boundary_squared_reg[eidx] = calc_regressions(exdict[ev])#dicts,evid)
#         deep_anomaly_rel_to_mld_min[eidx], deep_anomaly_rel_to_mld_mean[eidx], deep_anomaly_rel_to_mld_median[eidx] = calculate_depth_anomaly_rel_to_ml(exdict[ev])
#     chars = dict()
#     chars['duration'] = dict()
#     chars['duration']['column']=duration_c
#     chars['duration']['surface']=duration_s
#     chars['starttime'] = dict()
#     chars['starttime']['column']=starttime_c
#     chars['starttime']['surface']=starttime_s
#     chars['endtime'] = dict()
#     chars['endtime']['column']=endtime_c
#     chars['endtime']['surface']=endtime_s
#     chars['ietas']=etas
#     chars['ixis']=xis
#     chars['times']=times    
#     chars['areas']=areavec
#     chars['collevents_surf']=collected_events
#     chars['maxgap_surf']=maxgap
#     chars['maxconsec_surf']=maxconsecsurf
#     chars['surf10time']=surf10time
#     chars['surf90time']=surf90time
#     chars['maxdepth_abs']=maxdepth_abs
#     chars['meandepth_abs']=meandepth_abs
#     chars['mediandepth_abs']=mediandepth_abs
#     chars['fraction_in_ml']=fraction_in_ml
#     chars['max_fraction_in_ml']=max_fraction_in_ml
#     chars['mean_fraction_in_ml']=mean_fraction_in_ml
#     chars['median_fraction_in_ml']=median_fraction_in_ml
#     chars['upper_boundary_linear_reg']=upper_boundary_linear_reg
#     chars['upper_boundary_squared_reg']=upper_boundary_squared_reg
#     chars['deep_anomaly_rel_to_mld_min'] = deep_anomaly_rel_to_mld_min
#     chars['deep_anomaly_rel_to_mld_mean'] = deep_anomaly_rel_to_mld_mean
#     chars['deep_anomaly_rel_to_mld_median'] = deep_anomaly_rel_to_mld_median
#     return chars

# def set_up_L3D_char_structure(struct_type):
#     if struct_type == 'glsc':
#         characteristic_struct = dict()
#         characteristic_struct['global'] = dict()
#         characteristic_struct['local'] = dict()
#         characteristic_struct['global']['column'] = [] #overall duration of the event scalar value per event 
#         characteristic_struct['local']['surface'] = [] # how long an event affects a single surface grid cell (local)
#         characteristic_struct['local']['column'] = [] # how long an event affects a single horizontal grid cell (local)
#         characteristic_struct['global']['surface'] = [] # how long an event affects any surface grid cell (local)
#     elif struct_type == 'gl':
#         characteristic_struct = dict()
#         characteristic_struct['global'] = []
#         characteristic_struct['local'] = []  
#     elif struct_type == 'gsc':
#         characteristic_struct = dict()
#         characteristic_struct['global'] = dict()
#         characteristic_struct['global']['column'] = []     
#         characteristic_struct['global']['surface'] = []     
#     return characteristic_struct

# import time
# def get_chars_core_L3D(exdict,gridboxvolume,mld):
#     # all local characteristics are sorted according to the footprint indices
#     chars = dict()
#     chars['duration'] = set_up_L3D_char_structure('glsc')
#     chars['maxdepth_abs'] = set_up_L3D_char_structure('gl')
#     chars['fraction_in_ml'] = set_up_L3D_char_structure('gl')
#     chars['max_fraction_in_ml']=set_up_L3D_char_structure('gl')
#     chars['mean_fraction_in_ml']=set_up_L3D_char_structure('gl')
#     chars['median_fraction_in_ml']=set_up_L3D_char_structure('gl')
#     chars['footprint_area']=set_up_L3D_char_structure('gsc')
#     chars['max_footprint_area']=set_up_L3D_char_structure('gsc')
#     chars['mean_footprint_area']=set_up_L3D_char_structure('gsc')
#     chars['median_footprint_area']=set_up_L3D_char_structure('gsc')
#     chars['collevents_surf'] = []
#     chars['maxgap_surf'] = []
#     chars['maxconsec_surf'] = []
#     chars['ietas_surf']=[]
#     chars['ixis_surf']=[]
#     chars['ietas_footprint']=[]
#     chars['ixis_footprint']=[]
#     chars['itimes_footprint']=[]
#     chars['volume'] = []
#     chars['maxvolume'] = []
#     chars['meanvolume'] = []
#     chars['medianvolume'] = [] 
#     chars['starttime_global'] = []
#     chars['endtime_global'] = []
#     for eidx,ev in enumerate(exdict.keys()):
#         if np.mod(eidx,1000)==0:
#             print(eidx,ev)
#         chars['starttime_global'].append(np.min(np.array(exdict[ev]['itime'])))
#         chars['endtime_global'].append(np.max(np.array(exdict[ev]['itime'])))
#         # set up the coordinate array for each event
#         start = time.time()
#         stacked_locs = np.stack((exdict[ev]['idepth'],exdict[ev]['ieta'],exdict[ev]['ixi'],exdict[ev]['itime']))
#         #m1 = time.time()
#         #print('M1')
#         #print(m1-start)
#         #print(stacked_locs)
#         ## Calculate the global event duration    
#         chars['duration']['global']['column'].append(np.max(exdict[ev]['itime'])-np.min(exdict[ev]['itime'])+1)
#         ## calculate the global event surface duration
#         uniquer = exdict[ev]['idepth']==0
#         chars['duration']['global']['surface'].append(np.size(np.unique(exdict[ev]['itime'][uniquer])))
#         # calculate the global maximum depth
#         chars['maxdepth_abs']['global'].append(z[np.max(exdict[ev]['idepth'])])      
#         ## Calculate the local duration (i.e. at individual locations, but irrespective of depth)
#         footprint_dum = np.unique(stacked_locs[1:,:],axis=1)         # find unique horizontal locations
#         footprint,daycount = np.unique(footprint_dum[:-1,:],axis=1,return_counts=True) # find number of days at each horizontal location 
#         chars['ietas_footprint'].append(footprint[0,:])
#         chars['ixis_footprint'].append(footprint[1,:])
#         chars['duration']['local']['column'].append(daycount) 
#         #m2 = time.time()
#         #print('M2')
#         #print(m2-m1)
#         # calculate the max depth locally
#         local_maxdepth_dum = np.zeros_like(footprint[0,:])
#         print(np.shape(footprint))
#         timeis = []
#         lati = stacked_locs[1,:]
#         loni = stacked_locs[2,:]
#         timmi = stacked_locs[-1,:]
#         deppi = stacked_locs[0,:]
#         for fp in range(np.shape(footprint)[1]):
#             #print(fp)
#             #locator = np.where((stacked_locs[1,:]==footprint[0,fp])*(stacked_locs[2,:]==footprint[1,fp]))[0]
#             locator = (lati==footprint[0,fp])*(loni==footprint[1,fp])
#             timeis.append(np.unique(timmi[locator]))
#             depthis = deppi[locator]
#             local_maxdepth_dum[fp] = np.array(z[np.max(depthis)])
#         #m3 = time.time()
#         #print('M3')
#         #print(m3-m2)
#         chars['itimes_footprint'].append(timeis)
#         chars['maxdepth_abs']['local'].append(local_maxdepth_dum)
#         ## calculate the local surface duration (i.e. at individual locations)
#         #print('M31')
#         #m31 = time.time()
#         #print(m31-m3)       
#         stacked_locs_surf = stacked_locs[1:,(np.where(stacked_locs[0,:]==0))[0]] # only select the locations and instances where the extreme is at the surface and exclude the depth row
#         #print('M32')
#         #m32 = time.time()
#         #print(m32-m31) 
#         loc,count = np.unique(stacked_locs_surf[:-1,:],axis=1,return_counts=True) # exclude the time row to identify uniques
#         latloncount = np.vstack((loc,count)) # array have three rows: (ieta,ixi,#days with surface expression) # exclude 
#         chars['ietas_surf'].append(loc[0,:])
#         chars['ixis_surf'].append(loc[1,:])
#         chars['duration']['local']['surface'].append(count) # list of arrays. 
#         ## Calculate the number of collected events at the surface for each location
#         #m4 = time.time()
#         #print('M4')
#         #print(m4-m3)
#         colls = np.zeros_like(count)
#         maxgaps = np.zeros_like(count)
#         maxconsecs = np.zeros_like(count)
#         for locidx in range(np.shape(loc)[1]):
#             cond = (stacked_locs_surf[0,:]==loc[0,locidx])*(stacked_locs_surf[1,:]==loc[1,locidx])
#             timesteps = np.array(stacked_locs_surf[-1,cond])
#             diffs = np.diff(timesteps)
#             collected_events = np.sum(diffs>1)+1
#             if len(diffs)>0:
#                 maxgap=np.max(diffs)-1
#                 maxconsec = calc_max_consec_ones(diffs)
#             else:
#                 maxgap=0
#                 maxconsec = 1
#             colls[locidx] = collected_events
#             maxgaps[locidx] = maxgap   
#             maxconsecs[locidx] = maxconsec
#         #m5 = time.time()
#         #print('M5')
#         #print(m5-m4)
#         chars['collevents_surf'].append(colls)  #  the number of discrete events occurring at each surface location affected by extreme
#         chars['maxgap_surf'].append(maxgaps)
#         chars['maxconsec_surf'].append(maxconsecs)
#         ## Calculate the volume 
#         # take stacked_locs array and extract for each given coordinate index combination the grid point volume
#         # the sum up the volumes with the same time step
#         # obtain a time series for each extreme describing the evolution of its volume
#         #m6 = time.time()
#         #print('M6')
#         #print(m6-m5)
#         voldum = []
#         for i in range(np.shape(stacked_locs)[1]):
#             dd = stacked_locs[0,i]
#             la = stacked_locs[1,i]
#             lo = stacked_locs[2,i]
#             voldum.append(gridboxvolume[dd,la,lo]) 
#         volarray = np.array(voldum)
#         unique_times = np.unique(stacked_locs[-1,:])
#         volume_timeseries = np.zeros_like(unique_times)
#         for i in range(len(unique_times)):
#             volume_timeseries[i] = np.sum(volarray[stacked_locs[-1,:]==unique_times[i]])
#         chars['volume'].append(volume_timeseries)
#         chars['maxvolume'].append(np.max(volume_timeseries))
#         chars['meanvolume'].append(np.mean(volume_timeseries))
#         chars['medianvolume'].append(np.median(volume_timeseries))
#         #m7 = time.time()
#         #print('M7')
#         #print(m7-m6)
#         # calculate the fraction of the extreme in the ML
#         in_ml = np.zeros_like(exdict[ev]['itime'])
#         dzs = np.zeros_like(exdict[ev]['itime'])
#         for i in range(len(exdict[ev]['itime'])):
#             mld_value = (-1*mld[exdict[ev]['itime'][i],exdict[ev]['ieta'][i],exdict[ev]['ixi'][i]])
#             in_ml[i] = 1*(z[exdict[ev]['idepth'][i]] > mld_value)
#             dzs[i] = dz[exdict[ev]['idepth'][i]]
#         # do it for local
#         fraction_in_ml_for_this_event = []
#         #print(chars['ietas_footprint'])
#         #print(chars['ixis_footprint'])
#         #m8 = time.time()
#         #print('M8')
#         #print(m8-m7)
#         timmi = stacked_locs[-1,:]
#         lati = stacked_locs[1,:]
#         loni = stacked_locs[2,:]
#         print(np.shape(footprint))
#         for fp in range(np.shape(footprint)[1]):
#             #m81 = time.time()
#             #print('loop for footprint location '+str(fp))
#             locator = (lati==footprint[0,fp])*(loni==footprint[1,fp])          
#             trues_array = np.zeros(len(timeis[fp]))
#             trues_array_meters = np.zeros(len(timeis[fp]))
#             #print(timeis[fp])
#             #print(np.size(timeis[fp]))
#             in_ml_local = in_ml[locator]
#             dz_local = dzs[locator]
#             timmi_local = timmi[locator]
#             for iidx,it in enumerate(timeis[fp]):
#                 #print('At location take the time steps: '+str(it))
#                 #print(it)
#                 in_mls = in_ml_local[timmi_local==it]
#                 dzs_cho = dz_local[timmi_local==it]
#                 #print(np.size(in_mls))
#                 #trues_array[iidx] = np.sum(in_mls)/np.size(in_mls)
#                 trues_array_meters[iidx] = np.sum(in_mls*dzs_cho)/np.sum(dzs_cho)
#                 #print(trues_array)
#             fraction_in_ml_for_this_event.append(trues_array_meters)
#             #m82 = time.time()
#             #print('M81')
#             #print((m82-m81)/len(timeis[fp]))              
#             #print(fraction_in_ml_for_this_event)
#         #m9 = time.time()
#         #print('M9')
#         #print(m9-m8)
#         chars['fraction_in_ml']['local'].append(fraction_in_ml_for_this_event)
#         # now do it for global (still yields a time series)
#         fraction_in_ml_global = []
#         for tim in np.unique(exdict[ev]['itime']):
#             #fraction = np.sum(in_ml[exdict[ev]['itime']==tim])/np.size(in_ml[exdict[ev]['itime']==tim])
#             fraction = np.sum(in_ml[exdict[ev]['itime']==tim]*dzs[exdict[ev]['itime']==tim])/np.sum(dzs[exdict[ev]['itime']==tim])
#             fraction_in_ml_global.append(fraction)
#         #m10 = time.time()
#         #print('M10')
#         #print(m10-m9)
#         chars['fraction_in_ml']['global'].append(fraction_in_ml_global)
#         # calculate the max, mean and median fraction_in_ml for local and global 
#         chars['max_fraction_in_ml']['global'].append(np.max(fraction_in_ml_global))
#         chars['mean_fraction_in_ml']['global'].append(np.mean(fraction_in_ml_global))
#         chars['median_fraction_in_ml']['global'].append(np.median(fraction_in_ml_global))
#         #print(fraction_in_ml_for_this_event)
#         maxfractions = []
#         meanfractions = []
#         medianfractions = []
#         #m11 = time.time()
#         #print('M11')
#         #print(m11-m10)
#         for vals in fraction_in_ml_for_this_event:
#             maxfractions.append(np.max(vals))
#             meanfractions.append(np.mean(vals))
#             medianfractions.append(np.median(vals))
#         chars['max_fraction_in_ml']['local'].append(maxfractions)
#         chars['mean_fraction_in_ml']['local'].append(meanfractions)
#         chars['median_fraction_in_ml']['local'].append(medianfractions)   
#         # calculate the footprint areas
#         area_surf = []
#         area_column = []
#         #m12 = time.time()
#         #print('M12')
#         #print(m12-m11)
#         for ttt in np.unique(stacked_locs_surf[-1,:]):
#             locat = stacked_locs_surf[-1,:]==ttt
#             locas = np.unique(stacked_locs_surf[:,locat],axis=1)
#             area_surf.append(np.sum(area[locas[0,:],locas[1,:]]))
#         for ttt in np.unique(stacked_locs[-1,:]):
#             locat = stacked_locs[-1,:]==ttt
#             locas = np.unique(stacked_locs[1:,locat],axis=1)
#             area_column.append(np.sum(area[locas[0,:],locas[1,:]]))
#         #m13 = time.time()
#         #print('M13')
#         #print(m13-m12)
#         chars['footprint_area']['global']['surface'].append(area_surf)
#         chars['footprint_area']['global']['column'].append(area_column)
#         chars['max_footprint_area']['global']['surface'].append(np.max(area_surf))
#         chars['mean_footprint_area']['global']['surface'].append(np.mean(area_surf))
#         chars['median_footprint_area']['global']['surface'].append(np.median(area_surf))
#         chars['max_footprint_area']['global']['column'].append(np.max(area_column))
#         chars['mean_footprint_area']['global']['column'].append(np.mean(area_column))
#         chars['median_footprint_area']['global']['column'].append(np.median(area_column))         
#         ## Missing properties
#         # propagation index (horizontal, vertical)
#     return chars

# def keep_only_surface_events(exdict):
#     exdict_surface = dict()
#     for key in exdict:
#         exdict_surface[key] = dict()
#         exnum = -1
#         #print(key)
#         #if key == 'L3D':
#         #    key2 = 'L3D_cut'
#         #else:
#         #    key2 = key
#         array_name = exdict[key][key]
#         print(key)
#         for evidx,ev in enumerate(array_name):
#             #if np.mod(exnum,10000)==0:
#             #    print(exnum)
#             #print(evidx)
#             if np.any(ev[1]==0):
#                 exnum += 1
#                 exdict_surface[key][str(exnum)]=dict()
#                 exdict_surface[key][str(exnum)]['itime']=ev[0]
#                 exdict_surface[key][str(exnum)]['idepth']=ev[1]
#                 exdict_surface[key][str(exnum)]['ieta']=ev[2]
#                 exdict_surface[key][str(exnum)]['ixi']=ev[3]   
#     return exdict_surface

# def keep_only_long_events(exdict_surface,minduration=5):
#     exdict_surface_long = dict()
#     for key in exdict_surface:
#         exdict_surface_long[key] = dict()
#         exnum = -1
#         #print(key)
#         #if key == 'L3D':
#         #    key2 = 'L3D_cut'
#         #else:
#         #    key2 = key
#         #array_name = exdict_surface[key][key]
#         print(key)
#         for evidx,ev in enumerate(exdict_surface[key]):
#             print(ev)
#             #if np.mod(exnum,10000)==0:
#             #    print(exnum)
#             #print(evidx)
#             if np.size(np.unique(exdict_surface[key][ev]['itime']))>=minduration: #np.any(ev[1]==0):
#                 exnum += 1
#                 exdict_surface_long[key][str(exnum)]=dict()
#                 exdict_surface_long[key][str(exnum)]['itime']=exdict_surface[key][ev]['itime']
#                 exdict_surface_long[key][str(exnum)]['idepth']=exdict_surface[key][ev]['idepth']
#                 exdict_surface_long[key][str(exnum)]['ieta']=exdict_surface[key][ev]['ieta']
#                 exdict_surface_long[key][str(exnum)]['ixi']=exdict_surface[key][ev]['ixi']   
#     return exdict_surface_long    

# def get_characteristics(exdict,gridboxvolume,mld):
#     chardict = dict()
#     for key in exdict:
#         print(key)
#         if key == 'E0D':
#             print("E0D")
#             chardict[key] = get_chars_core(exdict[key],mld)
#         elif key == 'L1D':
#             print('L1D')
#             chardict[key] = get_chars_core(exdict[key],mld)
#         elif key == 'L3D':
#             print('L3D')
#             chardict[key] = get_chars_core_L3D(exdict[key],gridboxvolume,mld)
#     return chardict

# def get_intensities(exdict,cfn):
#     print('Load intensities array')
#     cat_index = cfn.variables['cat_index_smoothed'].values
#     print('Start the calculation')
#     intensdict = dict()
#     for key in exdict:
#         intensdict[key] = dict()
#         print(key)
#         intensdict[key]['intensities'] = dict()
#         intensdict[key]['max_intens'] = dict()
#         intensdict[key]['mean_intens'] = dict()
#         for eidx,evnum in enumerate(exdict[key].keys()):
#             if np.mod(eidx,10000)==0:
#                 print(eidx,evnum)
#             t = exdict[key][evnum]['itime']
#             s = exdict[key][evnum]['idepth']
#             e = exdict[key][evnum]['ieta']
#             x = exdict[key][evnum]['ixi']
#             i = cat_index[t,s,e,x]
#             intensdict[key]['intensities'][evnum] = i
#             intensdict[key]['max_intens'][evnum] = np.max(i)
#             intensdict[key]['mean_intens'][evnum] = np.mean(i)
#     return intensdict