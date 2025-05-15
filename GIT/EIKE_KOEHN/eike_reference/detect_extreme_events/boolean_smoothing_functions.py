"""
author: Eike Köhn
date: Jan 18, 2024
description: This file contains a number of functions that can be used to smooth the extreme boolean arrays according to Hobday et al. 2016 and Koehn et al. 2024
"""

import numpy as np
import scipy.ndimage

#%%
def generate_boolean_smoothing_kernel(dimension_name='temporal'):#params,dimension_name='temporal'):
    if dimension_name == 'horizontal':     
        boolean_smoothing_kernel = np.ones((3,3,3,3))
        boolean_smoothing_kernel[:,[0,2],:,:] = False         ## turn of vertical connection
        boolean_smoothing_kernel[[0,2],:,:,:] = False         ## turn of temporal connection
    elif dimension_name == 'vertical':     
        boolean_smoothing_kernel = np.zeros((3,3,3,3))
        boolean_smoothing_kernel[1,:,1,1]=True                ## turn on vertical connection
    elif dimension_name == 'temporal':
        boolean_smoothing_kernel = np.zeros((3,3,3,3))
        boolean_smoothing_kernel[:,1,1,1]=True                ## turn on temporal connection
    return boolean_smoothing_kernel 

def pad_for_initial_closing(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def pad_for_initial_opening(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 1)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def do_morphing(params,unsmoothed_array):
    # run the smoothing commands depending on the employed methodology
    if params.boolean_smoothing_type == 'Koehn2024':
        smoothing_kernel = generate_boolean_smoothing_kernel(dimension_name='temporal')
        closing_iterations = 2
        opening_iterations = 2
        padder = 2
        unsmoothed_array_padded = np.pad(unsmoothed_array,padder,pad_for_initial_closing)
        closed_array_padded = scipy.ndimage.binary_closing(unsmoothed_array_padded,structure=smoothing_kernel,iterations=closing_iterations)
        smoothed_array_padded = scipy.ndimage.binary_opening(closed_array_padded,structure=smoothing_kernel,iterations=opening_iterations)
        smoothed_array = smoothed_array_padded[padder:-padder,padder:-padder,padder:-padder,padder:-padder]
    elif params.boolean_smoothing_type == 'Hobday2016':
        smoothing_kernel = generate_boolean_smoothing_kernel(dimension_name='temporal')
        opening_iterations = 2
        closing_iterations = 1
        padder = 2
        unsmoothed_array_padded = np.pad(unsmoothed_array,padder,pad_for_initial_opening)
        opened_array_padded = scipy.ndimage.binary_opening(unsmoothed_array_padded,structure=smoothing_kernel,iterations=opening_iterations)
        smoothed_array_padded = scipy.ndimage.binary_closing(opened_array_padded,structure=smoothing_kernel,iterations=closing_iterations)
        smoothed_array = smoothed_array_padded[padder:-padder,padder:-padder,padder:-padder,padder:-padder]
    elif params.boolean_smoothing_type == 'no_smoothing':
        smoothed_array = unsmoothed_array
    else:
        raise ValueError('params.boolean_smoothing_type is not well defined.')   
    return smoothed_array

def smooth_boolean_array(params, unsmoothed_array):
    print('Do morphological operations.')
    print('Smooth the boolean according to {}.'.format(params.boolean_smoothing_type))
    smoothed_array = do_morphing(params,unsmoothed_array)
    return smoothed_array
