""" Final Project Functions"""
import numpy as np
import xarray as xr # NetCDF4 file handling
import matplotlib.pyplot as plt

def load_wind_data(path_nc, level="100m"):
    """
    Load wind data from NetCDF and calculate wind speed from u and v components.

    Parameters:
        path_nc (str): Path to NetCDF4 file.
        level (str): Height level, e.g., '10m' or '100m'.

    Returns:
        xr.DataArray: Wind speed [m/s].
    """
    # Load the NetCDF4 file
    ds = xr.open_dataset(path_nc)

    # Extract u and v components wind data
    if level == "100m":
        u = ds["u100"]
        v = ds["v100"]
    elif level == "10m":
        u = ds["u10"]
        v = ds["v10"]
    else:
        raise ValueError("Unsupported height level. Use '10m' or '100m'.")
    
    # Calculate wind speed from u and v components
    wind_speed = np.sqrt(u**2 + v**2)
    
    return wind_speed

    

def plot_wind_speed_histogram(wind_speed, level="100m"):
    """
    Plot histogram of wind speed.

    Parameters:
        wind_speed (xr.DataArray): Computed wind speed.
        level (str): Height level (for title).
    """
    plt.figure(figsize=(10, 6))
    wind_speed.mean(dim=["latitude", "longitude"]).plot.hist(bins=50)
    plt.title(f"Wind Speed Histogram at {level}")
    plt.xlabel("Wind Speed [m/s]")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
