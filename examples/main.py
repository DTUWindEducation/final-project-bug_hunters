from pathlib import Path
import numpy as np 
import pandas as pd 
import xarray as xr 
import matplotlib.pyplot as plt 
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

# define grid points as outlined by assignment
grid_points = [(55.5, 7.75), (55.5, 8.), (55.75, 7.75), (55.75, 8.)]

# create list with names of datafiles 
data_list = [DATA_97_99,DATA_00_02, DATA_03_05, DATA_06_08, DATA_09_11, DATA_12_14, DATA_15_17, DATA_18_20, DATA_21_23]

# concatenate data to create a dataframe containting data for entire time-span  
WindData = wra.conc_data(data_list)

# plotting for each coordinate point 
for lat, lon in grid_points: 
    WindSpdDir = wra.compute_and_plot_wind_speed_direction_time_series(WindData,grid_points,lat,lon,10)

# interpolation within grid for a point specified by user 
# TODO: figure out how to add ValueError for inputs not of numerical values?
interpolation_lat = input("Select latitude within the defined grid (between 55.5 - 55.75). The input must be a numberical value: ")
interpolation_long = input("Select longitude within the defined grid (between 7.75 - 8.). The input must be a numberical value: ")

interp_coords = [interpolation_lat, interpolation_long]

WindSpdDir = wra.compute_and_plot_wind_speed_direction_time_series(WindData,grid_points,interp_coords[0],interp_coords[1],10)



