#%%
from pathlib import Path
import numpy as np 
import pandas as pd 
import xarray as xr 
import matplotlib.pyplot as plt 
#from windrose import WindroseAxes
#from matplotlib.pyplot import get_cmap
from matplotlib.patches import Patch
import WRA_Package as wra

FILE_PATH = Path(__file__)      # path to this file
FILE_DIR = FILE_PATH.parent.parent     # path to main folder 
DATA_DIR = FILE_DIR / 'inputs'
DATA_97_99 = DATA_DIR / '1997-1999.nc'
DATA_00_02 = DATA_DIR / '2000-2002.nc'
DATA_03_05 = DATA_DIR / '2003-2005.nc'
DATA_06_08 = DATA_DIR / '2006-2008.nc'

# define locations for each grid point as outlined by assignment
locations = [(55.5, 7.75), (55.5, 8.00), (55.75, 7.75), (55.75, 8.00)]

# specifying a location within the grid to interpolate for
interpolation_lat = 55.68
interpolation_long = 7.82
interp_coords = (interpolation_lat, interpolation_long)

# adding interpolation coordinates to the locations
locations.append(interp_coords)

# create list with names of datafiles 
data_list = [DATA_97_99,DATA_00_02, DATA_03_05, DATA_06_08]

# concatenate data to create a dataframe containting data for entire time-span  
WindData = wra.conc_data(data_list)


# specifying which heights data should be plotted for in the time series 
time_series_heights = [10, 100]

# plotting for each coordinate point 
for height in time_series_heights: 
    for lat, lon in locations: 
        wra.compute_and_plot_time_series(WindData,lat,lon,height)


# --- Extrapolate wind speed to a custom height ---
# specify referance height
reference_height = 10
# specify target height 
target_height = 97



# Extract the appropriate wind speed time series
if reference_height not in [10, 100]:
    raise ValueError("Reference height must be 10 or 100 m.")

# create data frame to append wind speeds at target height for each grid corner
ExtrapolatedWindSpeed = pd.DataFrame({'Time': WindData['valid_time']})

for lat, lon in locations[:-1]: 
    # call function to produce dataframe containing wind speed at each location
    WindSpdDir = wra.compute_and_plot_time_series(WindData,lat, lon, reference_height,display_figure=False)
    # Also compute wind speeds at 10m and 100m (for dynamic alpha calculation)
    WindSpdDir_10m = wra.compute_and_plot_time_series(WindData, lat, lon, 10, display_figure=False)
    WindSpdDir_100m = wra.compute_and_plot_time_series(WindData, lat, lon, 100, display_figure=False)
    
    # access dataframe and extract wind speed at location
    u_ref = WindSpdDir['speed']
    u_10 = WindSpdDir_10m['speed']
    u_100 = WindSpdDir_100m['speed']

    # Extrapolate using dynamic alpha
    extrapolated_speed = wra.extrapolate_wind_speed(
        u_ref=u_ref,
        u_10=u_10,
        u_100=u_100,
        z_ref=reference_height,
        z_target=target_height
    )

    # Add extrapolated speed to DataFrame
    ExtrapolatedWindSpeed[f'({lat},{lon})'] = extrapolated_speed


# use function to calculate wind spd and direction for location inside box at 10 m
WindSpdDirWeibull = wra.compute_and_plot_time_series(WindData,locations[-1][0], locations[-1][1], reference_height,display_figure=False)

WindSpdDirWeibull_10m = wra.compute_and_plot_time_series(WindData, locations[-1][0], locations[-1][1], 10, display_figure=False)
WindSpdDirWeibull_100m = wra.compute_and_plot_time_series(WindData, locations[-1][0], locations[-1][1], 100, display_figure=False)

# set reference speed to the wind speed data calculated for location inside box at 10 m
u_ref_weibull = WindSpdDirWeibull['speed']
u_10_weibull = WindSpdDirWeibull_10m['speed']
u_100_weibull = WindSpdDirWeibull_100m['speed']

# Extrapolate using dynamic alpha
extrapolated_speed_weibull = wra.extrapolate_wind_speed(
    u_ref=u_ref_weibull,
    u_10=u_10_weibull,
    u_100=u_100_weibull,
    z_ref=reference_height,
    z_target=target_height
)

# --- Fit Weibull distribution to extrapolated wind speed ---
shape, scale = wra.fit_weibull_distribution(extrapolated_speed_weibull)

print(f"\nWeibull distribution fitted parameters at {target_height} m:")
print(f"Shape (k): {shape:.3f}")
print(f"Scale (A): {scale:.3f}")

# --- Plot histogram with Weibull PDF overlay ---
fig, ax = wra.plot_wind_speed_with_weibull(extrapolated_speed_weibull, shape, scale, level=f"{target_height}m")
plt.show()

# # --- Wind rose plot ---
# wra.plot_wind_rose(WindSpdDir['direction'], WindSpdDir['speed'], num_bins=8)