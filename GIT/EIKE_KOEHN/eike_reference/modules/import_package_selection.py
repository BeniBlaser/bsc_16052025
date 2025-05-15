"""
Load functions and modules
author: Eike E. Koehn
date: Apr 6, 2023 
"""

#%% import functions

import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors
import cmocean as cm

import datetime
from datetime import date

import scipy as sp
import scipy.stats as spstats
import scipy.signal as spsig
from scipy.optimize import curve_fit

import pickle
import glob
import re
import itertools

#exec(open('../modules/define_netcdf_functions.py').read())
exec(open('../modules/misc_plotting_functions.py').read())
exec(open('../modules/get_scriptdir_scriptname.py').read())
# %%
