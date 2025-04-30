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
    data = xr.open_dataset(file_path, decode_timedelta=True)
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

def separate_data_by_year(WindData, year):
    """
    Separates the WindData DataFrame for a specific year.

    Parameters:
        WindData (pd.DataFrame): The input DataFrame containing wind data with a 'valid_time' column.
        year (int): The year to filter the data for.

    Returns:
        pd.DataFrame: A DataFrame containing data only for the specified year.
    """
    # Ensure 'valid_time' is a datetime object
    WindData['valid_time'] = pd.to_datetime(WindData['valid_time'])

    # Filter the data for the specified year
    filtered_data = WindData[WindData['valid_time'].dt.year == year]

    # Check if data exists for the year
    if filtered_data.empty:
        raise ValueError(f"No data found for the year {year}.")

    return filtered_data


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


class WindInterpolator:
    """
    A class to interpolation wind data over a predefined grid.

    Parameters:
        grid_points (list of tuples): List of (latitude, longitude) tuples defining the grid.
        dataframe (pd.DataFrame): DataFrame containing wind data for all grid points.
    """
    def __init__(self, grid_points, dataframe):
        self.grid_points = np.array(grid_points)
        self.dataframe = dataframe
        
        self.u10 = []
        self.v10 = []
        self.u100 = []
        self.v100 = []
        self.times = None
        
        for lat, lon in grid_points:
            point_data = self.dataframe.loc[
                (self.dataframe['latitude'] == lat) & 
                (self.dataframe['longitude'] == lon)
            ].sort_values('valid_time')
            
            self.times = point_data['valid_time'].values
            self.u10.append(point_data['u10'].values)
            self.v10.append(point_data['v10'].values)
            self.u100.append(point_data['u100'].values)
            self.v100.append(point_data['v100'].values)
    
    def interpolate(self, interp_lat, interp_lon):
        """
        Interpolate wind components to a target location.
        
        Returns:
            (u10_interp, v10_interp, u100_interp, v100_interp, times)
        """
        return ( griddata(self.grid_points, self.u10, (interp_lat, interp_lon), method='linear'),
            griddata(self.grid_points, self.v10, (interp_lat, interp_lon), method='linear'),
            griddata(self.grid_points, self.u100, (interp_lat, interp_lon), method='linear'),
            griddata(self.grid_points, self.v100, (interp_lat, interp_lon), method='linear'),
            self.times )
    
    

def compute_and_plot_time_series(dataFrame, latitude, longitude, height=10, display_figure=True):
    grid_points = [(55.5, 7.75), (55.5, 8.), (55.75, 7.75), (55.75, 8.)]
    
    if (latitude, longitude) in grid_points:

        winddata = dataFrame.loc[
            (dataFrame['latitude'] == latitude) & 
            (dataFrame['longitude'] == longitude)
        ]

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
        dir_rad = np.arctan2(-u, -v) 
        dir_deg = (dir_rad * 180 / np.pi) % 360
        dir_deg = dir_deg
        speed = np.sqrt(u**2 + v**2)
        speed = speed
        time = time_values

    else:
        # Interpolate using WindInterpolator
        if not (55.5 <= latitude <= 55.75 and 7.75 <= longitude <= 8.0):
            raise ValueError("Coordinates are outside the interpolation grid. Reselect coordinates within the interpolation grid.")
        
        interpolator = WindInterpolator(grid_points, dataFrame)
        u10_interp, v10_interp, u100_interp, v100_interp, time = interpolator.interpolate(latitude, longitude)
        
        # Assign interpolated values
        u_10 = u10_interp
        v_10 = v10_interp
        u_100 = u100_interp
        v_100 = v100_interp
        
        if height == 10:
            u = u10_interp
            v = v10_interp
        elif height == 100:
            u = u100_interp
            v = v100_interp
        else:
            raise ValueError("Invalid height entry. Enter either 10 or 100.")
        
        # Calculate wind spd and dir
        dir_rad = np.arctan2(-u, -v)
        dir_deg = (dir_rad * 180 / np.pi) % 360
        speed = np.sqrt(u**2 + v**2)
        time = time

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

    
def calculate_alpha_dynamic(u_10, u_100):
    """
    Calculate the wind shear exponent alpha dynamically.

    Parameters:
        u_10 (array-like): Wind speed at 10 m height [m/s]
        u_100 (array-like): Wind speed at 100 m height [m/s]

    Returns:
        alpha (array-like): Wind shear exponent
    """
    u_10 = np.array(u_10)
    u_100 = np.array(u_100)
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.log(u_100 / u_10) / np.log(100 / 10)
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
    return alpha


def extrapolate_wind_speed(u_ref, u_10, u_100, z_ref, z_target):
    """
    Extrapolate wind speed using dynamically calculated alpha.

    Parameters:
        u_ref (array-like): Wind speed at reference height [m/s]
        u_10 (array-like): Wind speed at 10 m [m/s]
        u_100 (array-like): Wind speed at 100 m [m/s]
        z_ref (float): Reference height [m]
        z_target (float): Target height [m]

    Returns:
        array-like: Wind speed at target height
    """
    alpha = calculate_alpha_dynamic(u_10, u_100)
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


def plot_wind_rose(direction, speed, num_bins = 6, label_interval = 30): 
    ax = WindroseAxes.from_ax() 
    ax.bar(direction,
            speed, 
            normed = True, 
            nsector = 12, 
            edgecolor = 'white', 
            opening = 1.0,
            bins = num_bins,)
    ax.set_legend(loc=(-0.08, -0.08),title='Wind Speed [m/s]')
    angles = np.arange(0, 360, label_interval)          # 0°, 30°, 60° … 330°
    #ax.set_thetagrids(angles, [f"{a}°" for a in angles])
    # ax.set_thetagrids(range(0,360,30),[f"{a}°" for a in angles])
    # ax.set_theta_zero_location('W', offset=-90)

def calculate_bin_probabilities(data, bins):
    """
    Calculate the probabilities of data falling within specified bins.

    Parameters:
        data (pd.Series): The wind speed data.
        bins (list): The edges of the bins (e.g., [0, 5, 10, 15, 20]).

    Returns:
        dict: A dictionary where keys are bin ranges and values are probabilities (percentages).
    """
    # Use pandas cut to categorize data into bins
    bin_counts = pd.cut(data, bins=bins, right=False).value_counts(sort=False)

    # Calculate total data points
    total_count = len(data)

    # Calculate probabilities (percentages) for each bin
    bin_probabilities = {f"[{interval.left}, {interval.right})": (count / total_count) * 100
                         for interval, count in bin_counts.items()}

    return bin_probabilities

def calculate_aep(bin_probabilities, power_per_bin):
    """
    Calculate the Annual Energy Production (AEP).

    Parameters:
        bin_probabilities (dict): A dictionary where keys are bin ranges and values are probabilities (percentages).
        power_per_bin (dict): A dictionary where keys are bin ranges and values are power values (e.g., kW).

    Returns:
        float: The calculated AEP.
    """
    # Ensure the bin ranges in both dictionaries match
    if set(bin_probabilities.keys()) != set(power_per_bin.keys()):
        raise ValueError("Bin ranges in probabilities and power values do not match.")

    # Calculate AEP using the formula
    aep = 8760 * sum((bin_probabilities[bin_range] / 100) * power_per_bin[bin_range] for bin_range in bin_probabilities)

    return aep