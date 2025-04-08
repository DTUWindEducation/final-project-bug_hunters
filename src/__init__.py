import numpy as np

import pandas as pd

import xarray as xr

def load_data(file_path): 
    wind_data = xr.open_dataset(file_path)
    return wind_data 



