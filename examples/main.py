#%%
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


# define locations for each grid point as outlined by assignment
locations = [(55.5, 7.75), (55.5, 8.00), (55.75, 7.75), (55.75, 8.00)]

# specifying a location within the grid to interpolate for
interpolation_lat = 55.68
interpolation_long = 7.82
interp_coords = (interpolation_lat, interpolation_long)

# adding interpolation coordinates to the locations
locations.append(interp_coords)

# create list with names of datafiles 
data_list = [DATA_97_99,DATA_00_02, DATA_03_05, DATA_06_08, DATA_09_11, DATA_12_14, DATA_15_17, DATA_18_20, DATA_21_23]

# concatenate data to create a dataframe containting data for entire time-span  
WindData = wra.conc_data(data_list)


# specifying which heights data should be plotted for in the time series 
time_series_heights = [10, 100]

# plotting for each coordinate point 
for height in time_series_heights: 
    for lat, lon in locations: 
        wra.compute_and_plot_wind_speed_direction_time_series(WindData,lat,lon,height)


#%%

# --- Extrapolate wind speed to a custom height ---
# specify referance height
reference_height = 10
# specify target height 
target_height = 97
# specify alpha
alpha = 0.1


# Extract the appropriate wind speed time series
if reference_height not in [10, 100]:
    raise ValueError("Reference height must be 10 or 100 m.")

#%%

ExtrapolatedWindSpeed = pd.DataFrame({'Time': WindData['valid_time']})

for lat, lon in locations[:-1]: 
    # call function to produce dataframe containing wind speed at each location
    WindSpdDir = wra.compute_and_plot_wind_speed_direction_time_series(WindData,lat, lon, reference_height,display_figure=False)

    # access dataframe and extract wind speed at location
    u_ref = WindSpdDir['speed']

    # Extrapolate speed at new height using function
    extrapolated_speed = wra.extrapolate_wind_speed(u_ref, reference_height, target_height, alpha)

    # Add speed at each location to dataframe 
    ExtrapolatedWindSpeed[f'({lat},{lon})'] = extrapolated_speed





# --- Fit Weibull distribution to extrapolated wind speed ---
shape, scale = wra.fit_weibull_distribution(extrapolated_speed)

print(f"\nWeibull distribution fitted parameters at {target_height} m:")
print(f"Shape (k): {shape:.3f}")
print(f"Scale (A): {scale:.3f}")

# --- Plot histogram with Weibull PDF overlay ---
fig, ax = wra.plot_wind_speed_with_weibull(extrapolated_speed, shape, scale, level=f"{target_height}m")
plt.show()



#%%
# call bin_wind_dir_data to preform binning of directions
dir_edges, spd_edges, H = wra.windrose_hist(WindSpdDir['direction'], WindSpdDir['speed'], 12)
wra.plot_windrose(dir_edges, spd_edges, H)


# %%
