#%%
from pathlib import Path
import numpy as np 
import pandas as pd 
import xarray as xr 
import matplotlib.pyplot as plt 
from windrose import WindroseAxes
from matplotlib.pyplot import get_cmap
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
# print("WindData Preview:") # TODO - remove this line when finished
# print(WindData.head()) # TODO - remove this line when finished


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
# fig, ax = wra.plot_wind_speed_with_weibull(extrapolated_speed_weibull, shape, scale, level=f"{target_height}m")
# plt.show()


# wra.plot_wind_rose(WindSpdDir['direction'], WindSpdDir['speed'], num_bins = 8)

#7

# Call the function to separate data for the year 2005
try:
    WindData_2005 = wra.separate_data_by_year(WindData, 2005)
    #print("WindData for the year 2005:")
    #print(WindData_2005.head())
except ValueError as e:
     #print(e)
     WindData_2005 = None

# Check if data for 2005 exists
if WindData_2005 is not None:
    # Specify the point within the grid for interpolation
    interpolation_lat = 55.68
    interpolation_long = 7.82
    reference_height = 10  # Reference height for interpolation
    target_height = 90     # Target height for extrapolation
    alpha = 0.1            # Power law exponent

    # Interpolate wind data for the given point
    WindSpdDir_2005 = wra.compute_and_plot_time_series(
        WindData_2005, interpolation_lat, interpolation_long, reference_height, display_figure=False
    ) 

    # Extract wind speed at the reference height
    u_ref_2005 = WindSpdDir_2005['speed']
    u_10_2005 = WindSpdDir_2005['u10']
    u_100_2005 = WindSpdDir_2005['u100']

    # Extrapolate wind speed to the target height (90 meters)
    extrapolated_speed_90m = wra.extrapolate_wind_speed(u_ref_2005, u_10_2005, u_100_2005, reference_height, target_height)

    # Print the extrapolated wind speed
    # print(f"\nExtrapolated wind speed at {target_height} meters for point ({interpolation_lat}, {interpolation_long}):")
    # print(extrapolated_speed_90m.head())

# Define the bins for wind speed
bins = [
    3, 4, 5, 6, 7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8, 9, 10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
]

# Call the function to calculate probabilities
bin_probabilities = wra.calculate_bin_probabilities(extrapolated_speed_90m, bins)

#  Print the probabilities per bin
# print("\nWind Speed Probabilities per Bin:")
# for bin_range, probability in bin_probabilities.items():
#     print(f"{bin_range}: {probability:.2f}%")

# Define the power per bin (from the screenshot)
power_per_bin = {
    "[3.0, 4.0)": 40.52, "[4.0, 5.0)": 177.67, "[5.0, 6.0)": 403.9, "[6.0, 7.0)": 737.59, "[7.0, 7.1)": 1187.18,
    "[7.1, 7.2)": 1239.25, "[7.2, 7.3)": 1292.52, "[7.3, 7.4)": 1347.32, "[7.4, 7.5)": 1403.26,
    "[7.5, 7.6)": 1460.7, "[7.6, 7.7)": 1519.64, "[7.7, 7.8)": 1580.17, "[7.8, 7.9)": 1642.11,
    "[7.9, 8.0)": 1705.76, "[8.0, 9.0)": 1771.17, "[9.0, 10.0)": 2518.55, "[10.0, 10.1)": 3448.38,
    "[10.1, 10.2)": 3552.14, "[10.2, 10.3)": 3657.95, "[10.3, 10.4)": 3765.12,
    "[10.4, 10.5)": 3873.93, "[10.5, 10.6)": 3984.48, "[10.6, 10.7)": 4096.58,
    "[10.7, 10.8)": 4210.72, "[10.8, 10.9)": 4326.15, "[10.9, 11.0)": 4443.4, "[11.0, 11.1)": 4562.5,
    "[11.1, 11.2)": 4683.42, "[11.2, 11.3)": 4806.16, "[11.3, 11.4)": 4929.93,
    "[11.4, 11.5)": 5000.92, "[11.5, 11.6)": 5000.16, "[11.6, 11.7)": 4999.98,
    "[11.7, 11.8)": 4999.96, "[11.8, 11.9)": 4999.98, "[11.9, 12.0)": 5000, "[12.0, 13.0)": 5000,
    "[13.0, 14.0)": 5000.01, "[14.0, 15.0)": 5000.01, "[15.0, 16.0)": 5000.02, "[16.0, 17.0)": 5000.02,
    "[17.0, 18.0)": 5000.03, "[18.0, 19.0)": 5000.02, "[19.0, 20.0)": 5000.04, "[20.0, 21.0)": 5000.02,
    "[21.0, 22.0)": 5000.05, "[22.0, 23.0)": 5000.01, "[23.0, 24.0)": 5000.01, "[24.0, 25.0)": 5000.04
}

# Call the function to calculate AEP
aep = wra.calculate_aep(bin_probabilities, power_per_bin)

# Print the calculated AEP
print(f"\nAnnual Energy Production (AEP) for NREL 5 MW for 2005: {aep:.2f} kWh")

# %%
