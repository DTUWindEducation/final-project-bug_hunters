#%%
from pathlib import Path
import numpy as np 
import pandas as pd 
import xarray as xr 
import matplotlib as plt 
import WRA_Package as wra

FILE_PATH = Path(__file__)      # path to this file
FILE_DIR = FILE_PATH.parent.parent     # path to main folder 
DATA_DIR = FILE_DIR / 'inputs'
DATA_97_99 = DATA_DIR / '1997-1999.nc'
DATA_00_02 = DATA_DIR / '2000-2002.nc'
DATA_03_05 = DATA_DIR / '2003-2005.nc'
DATA_06_08 = DATA_DIR / '2006-2008.nc'
DATA_09_11 = DATA_DIR / '2009-2011.nc'
DATA_12_14 = DATA_DIR / '2012-2014.nc'
DATA_15_17 = DATA_DIR / '2015-2017.nc'
DATA_18_20 = DATA_DIR / '2018-2020.nc'
DATA_21_23 = DATA_DIR / '2021-2023.nc'

wind_speed_with_spatial_dims, wind_dir, time, lat, lon = wra.load_data(DATA_00_02,100)

wind_speed = wind_speed_with_spatial_dims.mean(dim=["latitude", "longitude"]) # I want to come back and look into this further, this should maybe be a function?

wra.plot_wind_speed_histogram(wind_speed,100)

wra.plot_wind_time_series(wind_speed,time,100)



wind_speed_with_spatial_dims, wind_dir, time, lat, lon = wra.load_data(DATA_00_02,10)

wind_speed = wind_speed_with_spatial_dims.mean(dim=["latitude", "longitude"]) # I want to come back and look into this further, this should maybe be a function?

wra.plot_wind_speed_histogram(wind_speed,10)

wra.plot_wind_time_series(wind_speed,time,10)



