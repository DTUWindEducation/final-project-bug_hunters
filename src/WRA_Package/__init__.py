import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def load_data(file_path, level = 100): 
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

    time = wind_data['valid_time']
    lat = wind_data['latitude']
    lon = wind_data['longitude']

    if level == 100: 
        u = wind_data['u100']
        v = wind_data['v100']
    elif level == 10:
        u = wind_data['u10']
        v = wind_data['v10']
    else: 
        print('Invalid height. Enter height value of either 100 or 10')

    wind_speed = np.sqrt(u**2 + v**2)
    wind_dir = (np.degrees(np.arctan2(-u, -v)) + 360) % 360

    
    return wind_speed, wind_dir, time, lat, lon



def plot_wind_speed_histogram(wind_speed, level="100m"):
    """
    Plot histogram of wind speed.

    Parameters:
        wind_speed (xr.DataArray): Computed wind speed.
        level (str): Height level (for title).
    """
    plt.figure(figsize=(10, 6))
    wind_speed.plot.hist(bins=50)
    plt.title(f"Wind Speed Histogram at {level}")
    plt.xlabel("Wind Speed [m/s]")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_wind_time_series(wind_speed, time, level=100): 
    plt.figure(figsize=(12,6))
    plt.plot(time,wind_speed)
    plt.title(f"Wind Speed Time Series at {level} m")
    plt.xlabel('Time')
    plt.ylabel('Wind Speed [m/s]')
    plt.grid(True)
    plt.tight_layout
    plt.show()



