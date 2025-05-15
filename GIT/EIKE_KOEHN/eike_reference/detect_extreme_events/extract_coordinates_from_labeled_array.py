"""
With this file, I extract the indices for each extreme event from the labeled array.
For memory reasons, I loop through the horizontal dimensions. 
This wil NOT work if I connect if extremes are tracked in full 3D, because of the lateral connections.

author: Eike E Köhn
date: Apr 30, 2022
"""

#%% DEFINE THE SCRIPTNAME
scriptdir = '/home/koehne/Documents/publications/paper_future_simulations/scripts/detect_extreme_events/'
scriptname = 'extract_coordinates_from_labeled_array.py'

#%% LOAD THE PACKAGES
import xarray as xr
import numpy as np
from scipy.sparse import csr_matrix
from datetime import date
import pickle
import sys

#%% LOAD THE CASES CLASS
exec(open('/home/bblaser/scripts/bsc_beni/eike_reference/modules/define_cases_and_parameters.py').read())

#%% DEFINE FUNCTIONS

def set_up_sparse_indices_array(params):
    sparse_indices = dict()
    # add the attributes to xarray dataset
    sparse_indices['author'] = 'E. E. Koehn'
    sparse_indices['date'] = str(date.today())
    sparse_indices['scriptdir'] = scriptdir
    sparse_indices['scriptname'] = scriptname
    sparse_indices['casename'] = casename
    sparse_indices['case_description'] = 'Given by the following attributes:'
    # for pattribute in dir(params):
    #     if 'keep_only' not in pattribute:
    #         if not pattribute.startswith('__') and not callable(getattr(params,pattribute)):
    #             print(pattribute)
    #             print(eval('params.'+pattribute))
    #             sparse_indices['params.'+pattribute] = eval('params.'+pattribute)
    #         if not pattribute.startswith('__') and callable(getattr(params,pattribute)):
    #             print(pattribute)
    #             print(eval('params.'+pattribute+'()'))
    #             sparse_indices['params.'+pattribute+'()'] = eval('params.'+pattribute+'()')
    sparse_indices['description'] = 'This dictionary contains all indices for each extreme event. The keys labeled_E0D etc. contain lists of tuples. Each list entry, i.e. each tuple, represents one extreme with the time,depth,x,y indices stored in this direction.'
    return sparse_indices

def open_labeled_array_dataset(params,config,scenario,varia):
    labeled_array_directory = params.dir_labeled_arrays
    labeled_array_filename = params._fname_labeled_array()
    fn = xr.open_dataset(labeled_array_directory+labeled_array_filename)
    return fn

#def load_labeled_array(fn,labeled_array_type):
#    labeled_array_name = 'labeled_'+labeled_array_type
#    labeled_array = fn.variables[labeled_array_name].values[:,:,:,:]
#    return labeled_array

def estimate_chunksize(fn,labeled_array_type,chunksizelimit_in_gb=15,chunking_dimension=2):
    chunksizelimit_in_bits = chunksizelimit_in_gb*1024*1024*1024*8 # bits
    labeled_array_name = 'labeled_'+labeled_array_type
    array_shape = fn.variables[labeled_array_name].shape
    array_dtype = fn.variables[labeled_array_name].dtype
    array_size = fn.variables[labeled_array_name].size
    if array_dtype=='float64' or array_dtype=='int64':
        bitspervalue = 64
    elif array_dtype=='float32' or array_dtype=='int32':
        bitspervalue = 32
    else:
        bitspervalue = 32
    array_size_in_bits = array_size*bitspervalue
    number_of_chunks = int(np.ceil(array_size_in_bits/chunksizelimit_in_bits))
    dimensionlength = array_shape[chunking_dimension]       # chunk along the first horizontal dimension, i.e. dimension 2)
    chunksize = int(np.floor(dimensionlength/number_of_chunks))
    return chunksize

def chunkwise_indices_extraction(fn,labeled_array_type,chunksize,chunking_dimension=2):
    print('Get sparse indices.')
    list_of_indices = []
    labeled_array_name = 'labeled_'+labeled_array_type
    dimensionlength = fn.variables[labeled_array_name].shape[chunking_dimension]
    print('Chunk along axis {} (dimension length: {}) with a chunksize of {}'.format(chunking_dimension, dimensionlength, chunksize))
    for ki in np.arange(0,dimensionlength,chunksize):
        k = int(ki)
        print('Processing chunk: {}-{}'.format(k,k+chunksize))
        print('Load labeled array chunk.')
        assert(chunking_dimension==2)
        labeled_array_chunk = fn.variables[labeled_array_name][:,:,k:k+chunksize,:].values    # ATTENTION: IF CHUNK_DIMENSION IS NOT 2, THINGS BREAK
        print(np.shape(labeled_array_chunk))
        print('Get sparse indices.')
        a = get_indices_sparse(labeled_array_chunk)
        print('Now go through the detected extreme indices list and correct for chunking in the respective dimension.')
        for ai in a:
            if np.size(ai[0])>0:
                aj = list(ai)
                aj[2] = aj[2]+k   
                ak = tuple(aj)
                list_of_indices.append(ak)
        del labeled_array_chunk
    return list_of_indices

def get_indices_sparse(data):
    print('get_indices_sparse_function')
    M = compute_M(data)
    print('M computation done')
    print('Put indices into list.')
    index_list = [np.unravel_index(row.data, data.shape) for row in M]
    print('Done.')
    return index_list

def compute_M(data):
    print('compute_M_function')
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))[1:,...]

def save_sparse(params,sparse_indices,config,scenario,varia):
    print('Define saving location for sparse_indices dictionary.')
    event_coordinates_fullpath = define_saving_path(params,config,scenario,varia)
    print('Save the dictionary containing the event indices.')   # of lists, containing the event indices for E0D, L1D and L3Ds
    with open(event_coordinates_fullpath, 'wb') as f:           # savepath+'extreme_indices_fac'+str(fac)+'_'+array_name+'_baseperiod_{}-{}.pck'.format(base_period_start,base_period_end), 'wb') as f:
        pickle.dump(sparse_indices, f, pickle.HIGHEST_PROTOCOL)  # Pickle the 'data' dictionary using the highest protocol available.

def define_saving_path(params,config,scenario,varia):
    print('set the saving path')
    if not os.path.exists('{}{}'.format(params.dir_root_event_coordinates,config)):
        os.mkdir('{}{}'.format(params.dir_root_event_coordinates,config))
        print('{}{} created. '.format(params.dir_root_event_coordinates,config))
    else:
        print('{}{} exists already. Do nothing. '.format(params.dir_root_event_coordinates,config))
    if not os.path.exists('{}{}/{}'.format(params.dir_root_event_coordinates,config,scenario)):
        os.mkdir('{}{}/{}'.format(params.dir_root_event_coordinates,config,scenario))
        print('{}{}/{} created.'.format(params.dir_root_event_coordinates,config,scenario))
    else:
        print('{}{}/{} exists already. Do nothing.'.format(params.dir_root_event_coordinates,config,scenario))
    savepath = '/home/bblaser/scripts/data_beni/temp/labeled_arrays/' #'{}{}/{}/'.format(params.dir_root_event_coordinates,config,scenario)

    event_coordinates_fullpath =  savepath+params._fname_extracted_coordinates(varia)
    return event_coordinates_fullpath

def single_extraction(params,labeled_array_type,chunksizelimit_in_gb,chunking_dimension,config,scenario,varia):
    print('Open the labeled array dataset.')
    fn = open_labeled_array_dataset(params,config,scenario,varia)
    print('Estimate the chunksize to avoid too long waiting times and to avoid storage overload.')
    chunksize = estimate_chunksize(fn,labeled_array_type,chunksizelimit_in_gb,chunking_dimension)
    #print('Load labeled array.')
    #labeled_array = load_labeled_array(fn,labeled_array_type)
    print('Do the slicewise indices extraction.')
    list_of_indices = chunkwise_indices_extraction(fn,labeled_array_type,chunksize,chunking_dimension)
    return list_of_indices

def full_extraction(params,labeled_array_types,chunksizelimit_in_gb,chunking_dimension,config,scenario,varia):
    print(labeled_array_types)
    print('Set up sparse indices array.')
    sparse_indices = set_up_sparse_indices_array(params)
    for labeled_array_type in labeled_array_types:
        print('Do the indices extraction for the labeled_array_type: {}'.format(labeled_array_type))
        list_of_indices = single_extraction(params,labeled_array_type,chunksizelimit_in_gb,chunking_dimension,config,scenario,varia)
        print('Put list of indices into sparse_indices dictionary.')
        sparse_indices[labeled_array_type] = list_of_indices
    print('Save the sparse indices.')
    save_sparse(params,sparse_indices,config,scenario,varia)
    #del labeled_array
    del sparse_indices


#%% GET THE EVENT INDICES (be carful with the memory usage! This can easily exceed KRYO's memory... > 300GB)
#casedir = '/home/koehne/Documents/publications/paper2/scripts_clean/modules/cases/'
casedir = '/home/bblaser/scripts/bsc_beni/eike_reference/modules/cases/'
#casedir = '/home/koehne/Documents/publications/paper_future_simuluations/scripts/modules/cases/'

casenames = ['case00.yaml'] #['case00_reference.yaml']

labeled_array_types = ['E0D','L1D']

chunksizelimit_in_gb = 30
chunking_dimension = 2

config = 'roms_only' #'romsoc_fully_coupled'
scenario = 'present'#'ssp585'
varia = 'temp'

for casename in casenames:
    params = read_config_files(casedir+casename)
    full_extraction(params,labeled_array_types,chunksizelimit_in_gb,chunking_dimension,config,scenario,varia)

# %%
