"""
This script performs wind resource assessment (WRA) by
processing wind data from multiple NetCDF files,
computing wind speed and direction time series,
extrapolating wind speeds to custom heights, fitting Weibull
distributions, and calculating the Annual Energy Production (AEP)
for a specified wind turbine.

Steps:
1. Load and parse multiple provided NetCDF4 files.
2. Compute wind speed and wind direction time series at
    10 m and 100 m heights for the four provided locations.
3. Compute wind speed and wind direction time series for a
    given location inside the box bounded by the four locations.
4. Extrapolate wind speed to a custom height using the power law profile.
5. Fit a Weibull distribution to the extrapolated wind speed data.
6. Plot wind speed distribution (histogram vs. fitted Weibull distribution)
    at a given location (inside the box) and a given height
7. Plot wind rose diagrams that showes the frequencies of different wind
    direction at a given location (inside the box) and a given height.
7. Calculate the AEP for a specified wind turbine at a given location and year.
8. Calculate and display the dominant wind direction for the entire time span.
9. Plot the power curve for the NREL 5 MW turbine.

"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import WRA_Package as wra
from WRA_Package import load_data

# 1. Load and parse multiple provided netCDF4 files
# List to store all figures for final display
figures_to_show = []
# Define file paths for input data (multiple files)
FILE_PATH = Path(__file__)      # path to this file
FILE_DIR = FILE_PATH.parent.parent     # path to main folder
DATA_DIR = FILE_DIR / 'inputs'
DATA_97_99 = DATA_DIR / '1997-1999.nc'
DATA_00_02 = DATA_DIR / '2000-2002.nc'
DATA_03_05 = DATA_DIR / '2003-2005.nc'
DATA_06_08 = DATA_DIR / '2006-2008.nc'
TURBINE_DATA = DATA_DIR / 'NREL_Reference_5MW_126.csv'

# Load results to output directory
OUTPUT_DIR = FILE_DIR / 'outputs' / 'data_files_you_generate'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # ensure the folder exists

# 2. Compute wind speed and wind direction time series at 10 m and
# 100 m heights for the four provided locations.'

# 3. Compute wind speed and wind direction time series at
# 10 m and 100 m heights for a given location inside
# the box bounded by the four locations, such as the
# Horns Rev 1 site, using interpolation.'

# define locations for each grid point as outlined by assignment
locations = [(55.5, 7.75),
             (55.5, 8.00),
             (55.75, 7.75),
             (55.75, 8.00)]  # Four corners of the grid

# specifying a location within the grid to interpolate for
interpolation_lat = 55.68
interpolation_long = 7.82
interp_coords = (interpolation_lat, interpolation_long)

# adding interpolation coordinates to the locations
locations.append(interp_coords)

# create list with names of datafiles
data_list = [DATA_97_99, DATA_00_02, DATA_03_05, DATA_06_08]

# concatenate data to create a dataframe containting data for entire time-span
WindData = wra.conc_data(data_list)

# specifying which heights data should be plotted for in the time series
time_series_heights = [10, 100]

# plotting for each coordinate point
for height in time_series_heights:
    for lat, lon in locations:
        wra.compute_and_plot_time_series(WindData, lat, lon, height)

# 4. Compute wind speed time series at height z for the 
# four provided locations using power law profile.

# specify referance height
reference_height = 10

# specify target height
target_height = 97


# Extract the appropriate wind speed time series
if reference_height not in [10, 100]:
    raise ValueError("Reference height must be 10 or 100 m.")

# create data frame to append wind speeds at target height for each grid corner
ExtrapolatedWindSpeed = pd.DataFrame()
unique_times = ((WindData['valid_time'])
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True))

ExtrapolatedWindSpeed['time'] = unique_times


for lat, lon in locations:
    # Compute wind speeds at 10m and 100m (for dynamic alpha calculation)
    WindSpdDir_10m = wra.compute_and_plot_time_series(WindData,
                                                      lat,
                                                      lon,
                                                      10,
                                                      display_figure=False)

    WindSpdDir_100m = wra.compute_and_plot_time_series(WindData,
                                                       lat,
                                                       lon,
                                                       100,
                                                       display_figure=False)

    # access dataframe and extract wind speed at location
    # u_ref = WindSpdDir_10['speed']
    u_10 = WindSpdDir_10m['speed']
    u_100 = WindSpdDir_100m['speed']

    # Extrapolate using dynamic alpha
    extrapolated_speed = wra.extrapolate_wind_speed(
        u_ref=u_10,
        u_10=u_10,
        u_100=u_100,
        z_ref=reference_height,
        z_target=target_height
    )

    # Add extrapolated speed to DataFrame
    ExtrapolatedWindSpeed[f'Wind Spd ({lat},{lon})'] = extrapolated_speed.values


# Save CSV
csv_path = OUTPUT_DIR / 'extrapolated_wind_speeds.csv'
ExtrapolatedWindSpeed.to_csv(csv_path,
                             index=False,
                             sep=';',
                             float_format='%.2f')

# use function to calculate wind spd and
# direction for location inside box at 10 m
WindSpdDirWeibull = wra.compute_and_plot_time_series(WindData,
                                                     locations[-1][0],
                                                     locations[-1][1],
                                                     reference_height,
                                                     display_figure=False)

WindSpdDirWeibull_10m = wra.compute_and_plot_time_series(WindData,
                                                         locations[-1][0],
                                                         locations[-1][1],
                                                         10,
                                                         display_figure=False)

WindSpdDirWeibull_100m = wra.compute_and_plot_time_series(WindData,
                                                          locations[-1][0],
                                                          locations[-1][1],
                                                          100,
                                                          display_figure=False)

# set reference speed to the wind speed data calculated
# for location inside box at 10 m
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

# 5. Fit Weibull distribution for wind speed at a given
# location (inside the box) and a given height.

# Fit Weibull distribution to extrapolated wind speed
shape, scale = wra.fit_weibull_distribution(extrapolated_speed_weibull)

print(f"\nWeibull distribution fitted parameters at {target_height} m:")
print(f"Shape (k): {shape:.3f}")
print(f"Scale (A): {scale:.3f}")


# 6. Plot wind speed distribution (histogram vs. fitted Weibull distribution)
# at a given location (inside the box) and a given height.

# Plot histogram with Weibull PDF overlay
fig, ax = wra.plot_wind_speed_with_weibull(extrapolated_speed_weibull,
                                           shape,
                                           scale,
                                           locations[-1][0],
                                           locations[-1][1],
                                           level=f"{target_height}m")

figures_to_show.append(fig)

# Save figure in output file
fig_path_weibull = OUTPUT_DIR / f'weibull_fit_{target_height}m.png'
fig.savefig(fig_path_weibull)
figures_to_show.append(fig)

# appending direction to extrapolated wind speed values
ExtrapolatedWindSpeed['direction'] = WindSpdDir_10m['direction']

# 7. Plot wind rose diagram that showes the frequencies of
# different wind direction at a given location (inside the box)
# and a given height.

# create wind rose figure
wra.plot_wind_rose(
    ExtrapolatedWindSpeed['direction'],
    ExtrapolatedWindSpeed[
        f"Wind Spd ({interpolation_lat},"
        f"{interpolation_long})"
    ],
    locations[-1][0],
    locations[-1][1],
    target_height,
    num_bins=8)

# Call the function to separate data for the year 2005
try:
    WindData_2005 = wra.separate_data_by_year(WindData, 2005)
except ValueError:
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
        WindData_2005,
        interpolation_lat,
        interpolation_long,
        reference_height,
        display_figure=False
    ) 

    # Extract wind speed at the reference height
    u_ref_2005 = WindSpdDir_2005['speed']
    u_10_2005 = WindSpdDir_2005['u10']
    u_100_2005 = WindSpdDir_2005['u100']

    # Extrapolate wind speed to the target height (90 meters)
    extrapolated_speed_90m = wra.extrapolate_wind_speed(u_ref_2005,
                                                        u_10_2005,
                                                        u_100_2005,
                                                        reference_height,
                                                        target_height)

# Define the path to the NREL_Reference_5MW_126.csv file
NREL_FILE_PATH = Path("inputs/NREL_Reference_5MW_126.csv")

# Load the data using the load_data function
nrel_data = load_data(NREL_FILE_PATH)

# Extract wind speed values from the NREL data to define bins
bins = nrel_data['Wind Speed [m/s]'].tolist()

# Call the function to calculate probabilities
bin_probabilities = wra.calculate_bin_probabilities(extrapolated_speed_90m,
                                                    bins)

# Define the power per bin (from the screenshot)

# Load the data using the load_data function
nrel_data = load_data(NREL_FILE_PATH)

# Dynamically generate power_per_bin using the function from WRA_Package
power_per_bin = wra.generate_power_per_bin(nrel_data)

# Call the function to calculate AEP
aep = wra.calculate_aep(bin_probabilities, power_per_bin)

# 8. Compute AEP of a specifed wind turbine (NREL 5 MW or NREL 15 MW)
# at a given location inside the box for a given year in the
# period we have provided the wind data.'

# Print the calculated AEP
print(
    "\nAnnual Energy Production (AEP) for NREL 5 MW for 2005: "
    f"{aep:.2f} kWh"
)

# 9. Compute and print dominant wind direction'
# Dominant wind direction for interpolated location
lat, lon = locations[-1]

# call function to calculate dominant wind direction
dominant_dir, freq = wra.dominant_wind_direction(
    ExtrapolatedWindSpeed['direction']
    )
print(
    f"Dominant wind direction range: {dominant_dir} with {freq} occurrences.")

# 10. Plot power curve for NREL 5 MW turbine
# call function to plot power curve
wra.plot_power_curve(TURBINE_DATA)

# Show all collected figures at the end
plt.show()
