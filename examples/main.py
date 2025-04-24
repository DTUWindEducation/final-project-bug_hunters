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

# Print the WindData to check its contents
print("\n--- WindData Summary ---")
print(WindData)

# user input for height 
height = float(input("Select height at which wind speed and direction will be calculated. Enter either 10 or 100: "))

# plotting for each coordinate point 
for lat, lon in grid_points: 
    WindSpdDir = wra.compute_and_plot_wind_speed_direction_time_series(WindData,grid_points,lat,lon,height)

# interpolation within grid for a point specified by user 
# TODO: figure out how to add ValueError for inputs not of numerical values?
interpolation_lat = float(input("Select latitude within the defined grid (between 55.50 and 55.75). The input must be a numberical value: "))
interpolation_long = float(input("Select longitude within the defined grid (between 7.75 and 8.00). The input must be a numberical value: "))
interpolation_height = float(input("Select height at which wind speed and direction will be calculated. Enter either 10 or 100: "))

interp_coords = [interpolation_lat, interpolation_long]

WindSpdDir = wra.compute_and_plot_wind_speed_direction_time_series(WindData,grid_points,interp_coords[0],interp_coords[1],interpolation_height)

# --- Extrapolate wind speed to a custom height ---
reference_height = float(input("Enter reference height (either 10 or 100): "))
target_height = float(input("Enter target height to extrapolate to (e.g., 90 or 150): "))
alpha = float(input("Enter power law exponent alpha (default is 0.1): ") or 0.1)

# Extract the appropriate wind speed time series
if reference_height not in [10, 100]:
    raise ValueError("Reference height must be 10 or 100 m.")

# Filter original DataFrame to get wind speed at reference height
if reference_height == interpolation_height:
    u_ref = WindSpdDir["speed"]
else:
    raise ValueError("Reference height must match previously computed wind speed level.")

# Extrapolate to new height
extrapolated_speed = wra.extrapolate_wind_speed(u_ref, reference_height, target_height, alpha)

# Print extrapolated time series, just to check
#print(f"\nExtrapolated wind speed to {target_height} m (first 5 values):\n", extrapolated_speed.head())


# --- Fit Weibull distribution to extrapolated wind speed ---
shape, scale = wra.fit_weibull_distribution(extrapolated_speed)

print(f"\nWeibull distribution fitted parameters at {target_height} m:")
print(f"Shape (k): {shape:.3f}")
print(f"Scale (A): {scale:.3f}")

# --- Plot histogram with Weibull PDF overlay ---
fig, ax = wra.plot_wind_speed_with_weibull(extrapolated_speed, shape, scale, level=f"{target_height}m")
plt.show()

print(WindSpdDir)

# Call the function from WRA_Package
result = wra.count_directions_in_windrose_ranges(WindSpdDir)

# Print the result
print(result)

# Call the function to count wind directions in ranges
range_counts = wra.count_directions_in_windrose_ranges(WindSpdDir)

# Map the results to variables
count_0 = range_counts.get("0-45", 0)
count_45 = range_counts.get("45-90", 0)
count_90 = range_counts.get("90-135", 0)
count_135 = range_counts.get("135-180", 0)
count_180 = range_counts.get("180-225", 0)
count_225 = range_counts.get("225-270", 0)
count_270 = range_counts.get("270-315", 0)
count_315 = range_counts.get("315-360", 0)

# Print the counts for verification
print(f"Count 0-45: {count_0}")
print(f"Count 45-90: {count_45}")
print(f"Count 90-135: {count_90}")
print(f"Count 135-180: {count_135}")
print(f"Count 180-225: {count_180}")
print(f"Count 225-270: {count_225}")
print(f"Count 270-315: {count_270}")
print(f"Count 315-360: {count_315}")

# Example wind direction and speed data
wind_directions = [0, 45, 90, 135, 180, 225, 270, 315]
wind_speeds = [count_0, count_45, count_90, count_135, count_180, count_225, count_270, count_315]

# Call the plot_wind_rose function from WRA_Package
wra.plot_wind_rose(wind_directions, wind_speeds, height="10m")