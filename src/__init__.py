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

"""Define classes"""

class GeneralWindTurbine:
    def __init__(self, rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, name=None):
        self.rotor_diameter = rotor_diameter
        self.hub_height = hub_height
        self.rated_power = rated_power
        self.v_in = v_in
        self.v_rated = v_rated
        self.v_out = v_out
        self.name = name

    def get_power(self, wind_speed):
        """
        Compute power output [kW] for given wind_speed (scalar or numpy array).
        """
        wind_speed = np.array(wind_speed)

        power_output = np.zeros_like(wind_speed)

        # Region 2: cubic power law
        mask_ramp = (wind_speed >= self.v_in) & (wind_speed < self.v_rated)
        power_output[mask_ramp] = self.rated_power * (wind_speed[mask_ramp] / self.v_rated) ** 3

        # Region 3: constant power
        mask_flat = (wind_speed >= self.v_rated) & (wind_speed <= self.v_out)
        power_output[mask_flat] = self.rated_power

        # Other values remain zero

        return power_output
    
    class WindTurbine(GeneralWindTurbine):
    def __init__(self, rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, power_curve_data, name=None):
        super().__init__(rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, name)
        self.power_curve_data = power_curve_data  # numpy array: [:, 0] = wind speed, [:, 1] = power

    def get_power(self, wind_speed):
        """
        Compute power output using power curve data (interpolated).
        """
        wind_speed = np.array(wind_speed)
        power = np.interp(wind_speed, self.power_curve_data[:, 0], self.power_curve_data[:, 1], left=0, right=0)
        return power