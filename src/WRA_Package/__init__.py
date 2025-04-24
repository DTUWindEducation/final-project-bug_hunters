#%%
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import matplotlib.pyplot as plt
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


def plot_wind_time_series(df, level=10):
    fig, axs = plt.subplots(2,1,figsize=(12,6))
    axs[0].plot(df['time'],df['speed'])
    axs[0].set_title(f"Wind Speed Time Series at {level} m")
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Wind Speed [m/s]')
    axs[0].grid(True)

    axs[1].plot(df['time'],df['direction'])
    axs[1].set_title(f"Wind Direction Time Series at {level} m")
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Wind Direction [deg]')
    axs[1].grid(True)
    fig.tight_layout()
    return fig, axs



def compute_and_plot_wind_speed_direction_time_series(dataFrame, grid_points, latitude, longitude, height=10):
    # Check if the point is exactly on a grid point
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
        if not (55.5 <= latitude <= 55.75 and 7.75 <= longitude <= 8.0):
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
                u_values.append(point_data['u10'].values) #for all points - so it can be used for interpolation 
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

    plot_wind_time_series(df, height)
    
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

"""Part seven"""

#################################################

def count_directions_in_windrose_ranges(df):
    """
    Count how many rows in the DataFrame have 'direction' in specific wind rose ranges.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'direction' column.

    Returns:
        dict: A dictionary with counts for each wind rose range.
    """
    # Ensure the 'direction' column exists
    if 'direction' not in df.columns:
        raise ValueError("The DataFrame must contain a 'direction' column.")

    # Define the wind rose ranges
    ranges = [
        (0, 45),
        (45, 90),
        (90, 135),
        (135, 180),
        (180, 225),
        (225, 270),
        (270, 315),
        (315, 360)
    ]

    # Count rows in each range
    range_counts = {}
    for lower, upper in ranges:
        count = ((df['direction'] >= lower) & (df['direction'] < upper)).sum()
        range_counts[f"{lower}-{upper}"] = count

    return range_counts

################################

def plot_wind_rose(wind_directions, wind_speeds, height="10m"):
    """
    Plot a wind rose diagram where the length of the bars represents wind speed.

    Parameters:
        wind_directions (array-like): Wind direction data in degrees (0-360).
        wind_speeds (array-like): Wind speed data in m/s.
        height (str): Height level for the title (e.g., "10m" or "100m").
    """
    from windrose import WindroseAxes
    import matplotlib.pyplot as plt

    # Create a wind rose plot
    fig = plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax(fig=fig)
    
    # Plot the wind rose with wind speeds determining the bar lengths
    ax.bar(wind_directions, wind_speeds, normed=False, opening=0.8, edgecolor='white')

    # Add labels and title
    ax.set_title(f"Wind Rose Diagram at {height}", fontsize=14)
    ax.set_legend(title="Wind Speed [m/s]", loc="lower right", fontsize=10)
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
    ax.plot(x, weibull_pdf, 'r-', lw=2, label='Fitted Weibull PDF')

    ax.set_title(f"Wind Speed Distribution & Weibull Fit at {level}")
    ax.set_xlabel("Wind Speed [m/s]")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig, ax


# %%
