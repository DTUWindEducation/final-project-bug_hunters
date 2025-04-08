from pathlib import Path
import numpy as np 
import pandas as pd 
import xarray as xr 
import matplotlib as plt 
import src

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

data = src.load_data(DATA_00_02)
