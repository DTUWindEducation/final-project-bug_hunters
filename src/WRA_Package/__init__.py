
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


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
    data = xr.open_dataset(file_path)
    df = data.to_dataframe().reset_index()

    # changing names of lat and long columns since they are incorrectly labeled
    df = df.rename(columns={'latitude':'_tmp','longitude':'latitude'})
    df = df.rename(columns={'_tmp':'longitude'})
    return df


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
    df_concat = []
    for file in files_list:
        # assume load_data returns a DataFrame with exactly these columns:
        # ["valid_time","latitude","longitude","u10","v10","u100","v100"]
        df = load_data(file)  
        df_concat.append(df)

    # one-shot concatenation
    ConcatDF = pd.concat(df_concat, axis=0, ignore_index=True)
    return ConcatDF


def plot_wind_time_series(df, lat, lon, level=10):
    fig, axs = plt.subplots(2,1,figsize=(12,6))
    axs[0].plot(df['time'],df['speed'])
    axs[0].set_title(f"Wind Speed Time Series at {level} m [{lat}° N,{lon}° E]")
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Wind Speed [m/s]')
    axs[0].grid(True)

    axs[1].plot(df['time'],df['direction'])
    axs[1].set_title(f"Wind Direction Time Series at {level} m [{lat}° N,{lon}° E]")
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Wind Direction [deg]')
    axs[1].grid(True)
    fig.tight_layout()
    plt.show()
    return fig, axs



def compute_and_plot_wind_speed_direction_time_series(dataFrame, grid_points, latitude, longitude, height=10):
    # ensuring that lat and lon are floats 
    latitude = float(latitude)
    longitude = float(longitude)

    # check if the point is exactly on a grid point
    if (latitude, longitude) in grid_points:
        winddata = dataFrame.loc[(dataFrame['latitude'] == latitude) & 
                                 (dataFrame['longitude'] == longitude)]
        if height == 10:
            u = winddata['u10']
            v = winddata['v10']
            time_values = winddata['valid_time']
        elif height == 100:
            u = winddata['u100']
            v = winddata['v100']
            time_values = winddata['valid_time']
        else:
            raise ValueError("Invalid height entry. Enter either 10 or 100.")
        dir_rad = np.arctan2(-u, -v)  # Note: Using arctan2 directly for correct quadrant
        dir_deg = (dir_rad * 180 / np.pi) % 360
        dir_deg = dir_deg
        speed = np.sqrt(u**2 + v**2)
        speed = speed
        time = time_values

    else:
        # Interpolate for points within the grid
        if not (55.50 <= latitude <= 55.75 and 7.75 <= longitude <= 8.00):
            raise ValueError("Coordinates are outside the interpolation grid.")
        
        # Prepare data for interpolation
        points = []
        u_values = []
        v_values = []
        time_values = []
        
        for lat, lon in grid_points:
            point_data = dataFrame.loc[(dataFrame['latitude'] == lat) & 
                                      (dataFrame['longitude'] == lon)]
            points.append([lat, lon])
            if height == 10:
                u_values.append(point_data['u10'].values)
                v_values.append(point_data['v10'].values)
                time_values.append(point_data['valid_time'].values)
            elif height == 100:
                u_values.append(point_data['u100'].values)
                v_values.append(point_data['v100'].values)
                time_values.append(point_data['valid_time'].values)
            else: 
                raise ValueError("Invalid height entry. Enter either 10 or 100.")
        
        # Convert to numpy arrays
        points = np.array(points)
        u_values = np.array(u_values)
        v_values = np.array(v_values)
        time_values = np.array(time_values)
        
        # Perform interpolation using griddata
        u_interp = griddata(points, u_values, (latitude, longitude), method='linear')
        v_interp = griddata(points, v_values, (latitude, longitude), method='linear')
        
        u = np.array([u_interp])
        v = np.array([v_interp])

    
        # Calculate wind direction and speed
        dir_rad = np.arctan2(-u, -v)  # Note: Using arctan2 directly for correct quadrant
        dir_deg = (dir_rad * 180 / np.pi) % 360
        dir_deg = dir_deg[0]
        speed = np.sqrt(u**2 + v**2)
        speed = speed[0]
        time = time_values[0]

    df = pd.DataFrame({'time': time, 'speed': speed, 'direction': dir_deg})

    fig, axs = plot_wind_time_series(df, latitude, longitude, height)
    
    return df, fig, axs



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


    
