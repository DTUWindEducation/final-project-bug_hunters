#%%
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def load_data(file_path):
    """
    Load wind data from NetCDF and calculate wind speed from u and v components.

    Parameters:
        path_nc (str): Path to NetCDF4 file.
        level (str): Height level, e.g., '10m' or '100m'.

    Returns:
        xr.DataArray: Wind speed [m/s]
        time: datetime
        lat: latitude values (float64)
        lon: longitude values (float64)
    """
    wind_data = xr.open_dataset(file_path)
    WindData = wind_data.to_dataframe().reset_index()
    return WindData


def conc_data(files_list):
    """
    Given a list of NetCDF files, load each into a pandas DataFrame
    and concatenate them all into one big DataFrame.

    Parameters
    ----------
    files_list : list of str
        Paths to your .nc files.

    Returns
    -------
    pd.DataFrame
        A single DataFrame with all the rows from each file.
    """
    WindDataConc = []
    for file in files_list:
        # assume load_data returns a DataFrame with exactly these columns:
        # ["valid_time","latitude","longitude","u10","v10","u100","v100"]
        WindData = load_data(file)  
        WindDataConc.append(WindData)

    # one-shot concatenation
    concDF = pd.concat(WindDataConc, axis=0, ignore_index=True)
    return concDF

def compute_wind_speed_time_series(dataframe, latitude, longitude, height = 10):
    if (latitude == 55.5 & longitude == 7.75) or (latitude == 55.5 & longitude == 8) or (latitude == 55.75 & longitude == 7.75) or (latitude == 55.75 & longitude == 8): 
        windspeed = dataframe.loc[(dataframe['latitude'] >= latitude) & (dataframe['longitude'] <= longitude)]
        if height == 10:
            u = windspeed['u10']
            v = windspeed['v10']
        elif height == 100: 
            u = windspeed['u100']
            v = windspeed['v100']
        else: 
            raise ValueError("Invalid height entry. Enter either 10 or 100")
        speed = np.sqrt(u**2 + v**2)
        time = windspeed['valid_time']






def time_series(wind_speed, latitude, longitude): 
    time_series = wind_speed.sel(latitude=latitude, longitude=longitude, method="nearest")


def plot_wind_speed_histogram(time_series_data, level="100m"):
    """
    Plot histogram of wind speed.

    Parameters:
        wind_speed (xr.DataArray): Computed wind speed.
        level (str): Height level (for title).
    """
    fig, axs = plt.subplots(figsize=(12,6))
    axs.hist(time_series_data, bins=50)
    axs.set_title(f"Wind Speed Histogram at {level}")
    axs.set_xlabel("Wind Speed [m/s]")
    axs.set_ylabel("Frequency")
    axs.grid(True)
    fig.tight_layout()
    return fig, axs


def plot_wind_time_series(time_series_data, time, level=100):
    fig, axs = plt.subplots(figsize=(12,6))
    axs.plot(time,time_series_data)
    axs.set_title(f"Wind Speed Time Series at {level} m")
    axs.set_xlabel('Time')
    axs.set_ylabel('Wind Speed [m/s]')
    axs.grid(True)
    fig.tight_layout()
    return fig, axs


# class GeneralWindTurbine():
#     def __init__(rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, name = None):
#         self.rotor_diameter = rotor_diameter
#         self.hub_height = hub_height
#         self.rated_power = rated_power
#         self.v_in = v_in
#         self.v_out = v_out 
#         self.v_rated = v_rated
#         self.name = name 
    
#     def get_power(): 


    



