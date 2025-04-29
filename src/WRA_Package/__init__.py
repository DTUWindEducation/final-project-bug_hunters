#%%
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from matplotlib.pyplot import get_cmap
from matplotlib.patches import Patch
from scipy.interpolate import griddata
from scipy.stats import weibull_min # Weibull distribution for wind speed



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



def compute_and_plot_time_series(dataFrame, latitude, longitude, height=10, display_figure = True):

    grid_points = [(55.5, 7.75), (55.5, 8.), (55.75, 7.75), (55.75, 8.)]
    # Check if the point is exactly on a grid point
    if (latitude, longitude) in grid_points:
        winddata = dataFrame.loc[(dataFrame['latitude'] == latitude) & 
                                 (dataFrame['longitude'] == longitude)]
        
        u_10 = winddata['u10']
        v_10 = winddata['v10']

        u_100 = winddata['u100']
        v_100 = winddata['v100']

        time_values = winddata['valid_time']

        if height == 10:
            u = u_10
            v = v_10
        elif height == 100:
            u = u_100
            v = v_100
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
        if not (55.5 <= latitude <= 55.75 and 7.75 <= longitude <= 8.0):
            raise ValueError("Coordinates are outside the interpolation grid.")
        
        # Prepare data for interpolation
        points = []
        u_values_10 = []
        v_values_10 = []
        u_values_100 = []
        v_values_100 = []        
        time_values = []
        
        for lat, lon in grid_points:
            point_data = dataFrame.loc[(dataFrame['latitude'] == lat) & 
                                      (dataFrame['longitude'] == lon)]
            points.append([lat, lon])
            u_values_10.append(point_data['u10'].values)
            v_values_10.append(point_data['v10'].values)

            u_values_100.append(point_data['u100'].values)
            v_values_100.append(point_data['v100'].values)

            time_values.append(point_data['valid_time'].values)
        
        # Convert to numpy arrays
        points = np.array(points)
        u_values_10 = np.array(u_values_10)
        v_values_10 = np.array(v_values_10)

        u_values_100 = np.array(u_values_100)
        v_values_100 = np.array(v_values_100)

        time_values = np.array(time_values)


        # Perform interpolation using griddata
        u_interp_10 = griddata(points, u_values_10, (latitude, longitude), method='linear')
        v_interp_10 = griddata(points, v_values_10, (latitude, longitude), method='linear')

        # Perform interpolation using griddata
        u_interp_100 = griddata(points, u_values_100, (latitude, longitude), method='linear')
        v_interp_100 = griddata(points, v_values_100, (latitude, longitude), method='linear')

        
        u_10 = u_interp_10
        v_10 = v_interp_10

        u_100 = u_interp_100
        v_100 = v_interp_100

        if height == 10: 
            u = np.array([u_interp_10])
            v = np.array([v_interp_10])
        elif height == 100: 
            u = np.array([u_interp_100])
            v = np.array([v_interp_100])
        else: 
            raise ValueError("Invalid height entry. Enter either 10 or 100.")

    
        # Calculate wind direction and speed
        dir_rad = np.arctan2(-u, -v)  # Note: Using arctan2 directly for correct quadrant
        dir_deg = (dir_rad * 180 / np.pi) % 360
        dir_deg = dir_deg[0]
        speed = np.sqrt(u**2 + v**2)
        speed = speed[0]
        time = time_values[0]

    df = pd.DataFrame({'time': time, 'speed': speed, 'direction': dir_deg, 'u10': u_10, 'v10': v_10, 'u100': u_100, 'v100': v_100})

    if display_figure == True: 
        fig, axs = plot_wind_time_series(df,latitude, longitude, height)
        return df, fig, axs
    else: 
        return df 
    



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

    
def extrapolate_wind_speed(u_ref, z_ref, z_target, alpha=0.1):
    """
    Extrapolate wind speed using the power law profile.

    Parameters:
        u_ref (array-like): Wind speed at reference height [m/s]
        z_ref (float): Reference height [m]
        z_target (float): Target height [m]
        alpha (float): Power law exponent (default 0.1 for offshore)

    Returns:
        array-like: Wind speed at target height
    """
    return u_ref * (z_target / z_ref) ** alpha


def fit_weibull_distribution(wind_speeds):
    """
    Fit Weibull distribution to wind speed data.

    Parameters:
        wind_speeds: Wind speed time series [m/s]

    Returns:
        shape (float): Weibull shape parameter k
        scale (float): Weibull scale parameter A
    """
    shape, loc, scale = weibull_min.fit(wind_speeds, floc=0)  # force location to 0
    return shape, scale

def plot_wind_speed_with_weibull(wind_speeds, shape, scale, level="100m"):
    fig, ax = plt.subplots(figsize=(12, 6))
    count, bins, _ = ax.hist(wind_speeds, bins=50, density=True, alpha=0.6, label='Wind Speed Histogram')

    x = np.linspace(min(bins), max(bins), 100)
    weibull_pdf = weibull_min.pdf(x, shape, loc=0, scale=scale)
    ax.plot(x, weibull_pdf, 'r-', lw=2, label = f" Fitted Weibull PDF \n k= {round(shape,2)}, A = {round(scale,2)}")

    ax.set_title(f"Wind Speed Distribution & Weibull Fit at {level}")
    ax.set_xlabel("Wind Speed [m/s]")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig, ax


def plot_wind_rose(direction, speed, num_bins = 6): 
    ax = WindroseAxes.from_ax()
    ax.bar(direction,
            speed, 
            normed = True, 
            nsector = 12, 
            edgecolor = 'white', 
            opening = 1.0,
            bins = num_bins)
    ax.set_legend()
