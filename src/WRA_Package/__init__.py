from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from scipy.interpolate import griddata
from scipy.stats import weibull_min  # Weibull distribution for wind speed


def load_data(file_path):
    """
    Load wind data from NetCDF and calculate wind speed
    from u and v components.

    Parameters:
        path_nc (str): Path to NetCDF4 file.
        level (str): Height level, e.g., '10m' or '100m'.

    Returns:
        xr.DataArray: Wind speed [m/s]
        time: datetime
        lat: latitude values (float64)
        lon: longitude values (float64)
    """
    # Check if the file is a NetCDF file
    if file_path.suffix.lower() == '.nc':
        # Open NetCDF file
        data = xr.open_dataset(file_path, decode_timedelta=True)
        # Convert to DataFrame and reset index
        df = data.to_dataframe().reset_index()
    # Check if the file is a CSV file
    elif file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path, sep=",")  # Load CSV file
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
        WindData (pd.DataFrame): The input DataFrame containing
        wind data with a 'valid_time' column.
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

    """
    Plot wind speed and wind direction time series for
    a specific location and height.

    This function generates two subplots:
    1. A time series plot of wind speed at the specified height.
    2. A time series plot of wind direction at the specified height.

    The plots are saved as a PNG file in the
    "outputs/data_files_you_generate" directory.

    Parameters:
        df (pd.DataFrame): A DataFrame containing the wind data.
                           It must include the following columns:
                           - 'speed': Wind speed values [m/s].
                           - 'direction': Wind direction values [degrees].
                           - 'time': Timestamps corresponding to the wind data.
        lat (float): Latitude of the location [degrees].
        lon (float): Longitude of the location [degrees].
        level (int, optional): Height level for the
        wind data [meters]. Default is 10 m.

    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the plots.
        axs (numpy.ndarray): An array of Axes objects for the subplots.

    Raises:
        ValueError: If the input DataFrame does not
        contain the required columns ('speed', 'direction', 'time').

    Notes:
        - The function saves the generated plot as a PNG
        file in the "outputs/data_files_you_generate" directory.
        - The file name is dynamically generated based
        on the latitude, longitude, and height level.
    """
    # Check if the required columns are present in the DataFrame
    if ('speed' not in df.columns) or ('direction' not in df.columns) or ('time') not in df.columns: 
        raise ValueError("Dataframe must contain speed, direction, and time")
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    # Plot wind speed against time
    axs[0].plot(df['time'], df['speed'])
    # Set the title for the wind speed plot
    axs[0].set_title(f"Wind Speed Time Series at {level} m [{lat}° N,{lon}° E]")
    # Label the x-axis
    axs[0].set_xlabel('Time')
    # Label the y-axis
    axs[0].set_ylabel('Wind Speed [m/s]')
    # Add a grid to the plot for better readability
    axs[0].grid(True)

    # Plot wind direction time series
    axs[1].plot(df['time'], df['direction'])
    # Set the title for the wind direction plot
    axs[1].set_title(f"Wind Direction Time Series at {level} m [{lat}° N,{lon}° E]")
    # Label the x-axis
    axs[1].set_xlabel('Time')
    # Label the y-axis
    axs[1].set_ylabel('Wind Direction [deg]')
    # Add a grid to the plot for better readability
    axs[1].grid(True)

    # Adjust layout to prevent overlapping of subplots
    fig.tight_layout()
    # Define the output directory for saving the plot
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "data_files_you_generate"
    # Generate a dynamic file name based on
    # latitude, longitude, and height level
    figurename = f"time_series_{lat}_{lon}_{level}.png"
    # Save the figure as a PNG file in the specified output directory
    fig.savefig(output_dir / figurename, format="png", bbox_inches='tight')
    # plt.show()
    return fig, axs


class WindInterpolator:
    """
    A class to interpolation wind data over a predefined grid.

    Parameters:
        grid_points (list of tuples): List of (latitude, longitude)
        tuples defining the grid.
        dataframe (pd.DataFrame): DataFrame containing wind data
        for all grid points.
    """
    def __init__(self, grid_points, dataframe):
        # Convert list of grid points to a
        # NumPy array for efficient computation
        self.grid_points = np.array(grid_points)
        # Store the input dataframe containing wind data
        self.dataframe = dataframe

        # Initialize lists to store wind components at 10m and 100m
        self.u10 = []
        self.v10 = []
        self.u100 = []
        self.v100 = []
        # Placeholder for storing time values
        self.times = None

        # Loop through each grid point (latitude, longitude)
        for lat, lon in grid_points:
            # Filter dataframe for the current grid
            # point and sort by valid_time
            point_data = self.dataframe.loc[
                (self.dataframe['latitude'] == lat) &
                (self.dataframe['longitude'] == lon)
            ].sort_values('valid_time')
            # Extract and store time values (assumed consistent across points)
            self.times = point_data['valid_time'].values
            # Append wind component values at 10m and 100m
            # heights to respective lists
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
        # Perform spatial interpolation of wind components at the given lat/lon
        return (griddata(self.grid_points,
                         self.u10,
                         (interp_lat, interp_lon),
                         method='linear'),
                griddata(self.grid_points,
                         self.v10,
                         (interp_lat, interp_lon),
                         method='linear'),
                griddata(self.grid_points,
                         self.u100,
                         (interp_lat, interp_lon),
                         method='linear'),
                griddata(self.grid_points,
                         self.v100,
                         (interp_lat, interp_lon),
                         method='linear'),
                self.times)


def compute_and_plot_time_series(dataFrame,
                                 latitude,
                                 longitude,
                                 height=10,
                                 display_figure=True):
    """
    Compute wind speed and direction time series for a
    location and height, with optional plotting.

    Parameters:
        dataFrame (pd.DataFrame): Wind data containing 'latitude',
        'longitude', 'u10', 'v10', 'u100', 'v100', 'valid_time'.
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        height (int, optional): Height level (10 or 100 m). Default is 10.
        display_figure (bool, optional): Whether to display
        and save the plot. Default is True.

    Returns:
        df (pd.DataFrame): DataFrame with 'time', 'speed',
        'direction', 'u10', 'v10', 'u100', 'v100'.
        fig, axs (optional): Plot figure and axes if `display_figure` is True.

    Raises:
        ValueError: If height is invalid or coordinates are outside the grid.
    """
    # Define a list of valid grid points as (latitude, longitude) tuples
    grid_points = [(55.5, 7.75), (55.5, 8.), (55.75, 7.75), (55.75, 8.)]
  
    # Check if the given latitude and longitude match any
    # of the defined grid points
    if (latitude, longitude) in grid_points:
        # Filter the DataFrame to only include rows matching
        # the specified latitude and longitude
        winddata = dataFrame.loc[
            (dataFrame['latitude'] == latitude) &
            (dataFrame['longitude'] == longitude)
        ]
        # Extract wind components at 10m height
        u_10 = winddata['u10']
        v_10 = winddata['v10']
        # Extract wind components at 100m height
        u_100 = winddata['u100']
        v_100 = winddata['v100']
        # Extract corresponding time values
        time_values = winddata['valid_time']
        # Select the appropriate wind components based on the given height
        if height == 10:
            u = u_10
            v = v_10
        elif height == 100:
            u = u_100
            v = v_100
        else:
            # Raise an error if height is not 10 or 100
            raise ValueError("Invalid height entry. Enter either 10 or 100.")
        # Compute wind direction in radians and convert to degrees
        dir_rad = np.arctan2(-u, -v) 
        dir_deg = (dir_rad * 180 / np.pi) % 360
        dir_deg = dir_deg

        # Compute wind speed from u and v components
        speed = np.sqrt(u**2 + v**2)
        speed = speed
        time = time_values

    else:
        # Interpolate using WindInterpolator
        if not (55.5 <= latitude <= 55.75 and 7.75 <= longitude <= 8.0):
            raise ValueError("Coordinates are outside the interpolation "
                             "grid. Reselect coordinates within "
                             "the interpolation grid.")

        interpolator = WindInterpolator(grid_points, dataFrame)
        u10_interp, v10_interp, u100_interp, v100_interp, time = interpolator.interpolate(latitude, longitude)

        # Assign interpolated values
        u_10 = u10_interp
        v_10 = v10_interp
        u_100 = u100_interp
        v_100 = v100_interp

        # Select the appropriate wind components based on the specified height
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

        # Calculate wind speed from u and v components
        speed = np.sqrt(u**2 + v**2)

        # Assign time values (interpolated or directly from the dataset)
        time = time

    # Create a DataFrame to store the computed
    # wind speed, direction, and components
    df = pd.DataFrame({'time': time,
                       'speed': speed,
                       'direction': dir_deg,
                       'u10': u_10,
                       'v10': v_10,
                       'u100': u_100,
                       'v100': v_100})

    # Check if the user wants to display and save the figur
    if display_figure == True:
        # Plot wind time series and return the figure and axes
        fig, axs = plot_wind_time_series(df, latitude, longitude, height)
        # Return the DataFrame, figure, and axes
        return df, fig, axs
    else:
        # Return only the DataFrame if no figure is requested
        return df


def calculate_alpha_dynamic(u_10, u_100):
    """
    Calculate the wind shear exponent alpha dynamically.

    Parameters:
        u_10 (array-like): Wind speed at 10 m height [m/s]
        u_100 (array-like): Wind speed at 100 m height [m/s]

    Returns:
        alpha (array-like): Wind shear exponent
    """
    # Convert wind speed arrays at 10m and 100m heights to
    # NumPy arrays for computation
    u_10 = np.array(u_10)
    u_100 = np.array(u_100)

    # Suppress warnings for invalid operations
    # (e.g., division by zero) during calculations
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate the wind shear exponent (alpha)
        # using the logarithmic wind profile formula
        alpha = np.log(u_100 / u_10) / np.log(100 / 10)

        # Replace NaN, positive infinity, and negative infinity values with 0.0
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

    # Return the calculated wind shear exponent (alpha)
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
    # Calculate the wind shear exponent (alpha) dynamically
    # based on wind speeds at 10m and 100m
    alpha = calculate_alpha_dynamic(u_10, u_100)

    # Extrapolate wind speed to the target height using the power law formula
    # The formula is: u_target = u_ref * (z_target / z_ref) ** alpha
    # where:
    # - u_ref is the wind speed at the reference height
    # - z_ref is the reference height
    # - z_target is the target height
    # - alpha is the wind shear exponent
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
    # Fit a Weibull distribution to the wind speed data
    # force location to 0
    shape, loc, scale = weibull_min.fit(wind_speeds, floc=0)
    # Return the shape (k) and scale (A) parameters of the Weibull distribution
    return shape, scale


def plot_wind_speed_with_weibull(wind_speeds,
                                 shape,
                                 scale,
                                 lat,
                                 lon,
                                 level="100m"):
    """
    Plot the wind speed distribution as a histogram
    and overlay the fitted Weibull PDF.

    Parameters:
        wind_speeds (array-like): Wind speed data [m/s].
        shape (float): Weibull shape parameter (k).
        scale (float): Weibull scale parameter (A).
        lat (float): latitude of point for which data is being plotted.
        lon (float): longitude of point for which data is being plotted.
        level (str, optional): Height level for the
        wind data (e.g., "100m"). Default is "100m".

    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the plot.
        ax (matplotlib.axes.Axes): The axes object for the plot.

    Notes:
        - The histogram shows the distribution of wind speeds.
        - The Weibull PDF is overlaid to visualize the fitted distribution.
        - The plot includes a legend and grid for better readability.
    """

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot a histogram of the wind speeds
    count, bins, _ = ax.hist(wind_speeds,
                             bins=50,
                             density=True,
                             alpha=0.6,
                             label='Wind Speed Histogram')

    # Generate x values for the Weibull PDF
    x = np.linspace(min(bins), max(bins), 100)

    # Calculate the Weibull PDF using the fitted parameters
    weibull_pdf = weibull_min.pdf(x, shape, loc=0, scale=scale)

    # Plot the Weibull PDF
    ax.plot(x,
            weibull_pdf,
            'r-',
            lw=2,
            label=(
                f" Fitted Weibull PDF \n"
                f" k= {round(shape, 2)}, A = {round(scale, 2)}"
                )
            )

    # Set the title and labels for the plot
    ax.set_title(
                "Wind Speed Distribution & Weibull"
                f"Fit at ({lat}°N {lon}°E) and {level}"
                )
    ax.set_xlabel("Wind Speed [m/s]")
    ax.set_ylabel("Probability Density")

    # Add a legend to the plot
    ax.legend()

    # Add a grid for better readability
    ax.grid(True)

    # Adjust the layout to prevent overlapping
    plt.tight_layout()

    # Return the figure and axis objects
    return fig, ax


def plot_wind_rose(direction,
                   speed,
                   lat,
                   lon,
                   height,
                   num_bins=6):
    """
    Plot a wind rose diagram showing the 
    distribution of wind direction and speed.

    Parameters:
        direction (array-like): Wind direction data in degrees [0, 360).
        speed (array-like): Wind speed data [m/s].
        lat (float): latitude coordinate of wind data.
        lon (float): longitude coordinate of wind data.
        height (float): height of wind data.
        num_bins (int, optional): Number of bins for
        wind speed categories. Default is 6.
        label_interval (int, optional): Interval for
        wind direction labels in degrees. Default is 30.

    Returns:
        None: Saves the wind rose plot as a PNG file in the output directory.

    Notes:
        - The wind rose divides wind direction into
            12 sectors (30° each) by default.
        - Frequencies are normalized to percentages.
        - The plot is saved in the "outputs/data_files_you_generate" directory.
    """
    # Create a new figure for the wind rose plot
    fig = plt.figure()

    # Create a WindroseAxes object for the wind rose plot
    ax = WindroseAxes.from_ax(fig=fig)

    # Plot the wind rose using wind direction and speed data
    ax.bar(
            direction,  # Wind direction data
            speed,  # Wind speed data
            normed=True,  # Normalize the frequencies to percentages
            nsector=12,  # Divide the wind direction into 12 sectors (30° each)
            edgecolor='white',  # Set the edge color of the bars to white
            opening=1.0,  # Set opening factor for the bars (1.0 = fully open)
            bins=num_bins,  # Number of bins for wind speed categories
        )

    # Add a legend to the wind rose plot
    ax.set_legend(loc=(-0.2, -0.1), title='Wind Speed [m/s]')

    # Set the title for the wind rose plot
    ax.set_title(f'Wind Rose, ({lat}°N {lon}°E) at {height} m')

    # Get the figure object from the WindroseAxes
    fig = ax.figure

    # Define the output directory for saving the wind rose plot
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "data_files_you_generate"

    # Generate a file name for the wind rose plot
    figurename = "wind_rose.png"

    # Save the wind rose plot as a PNG file in the specified output directory
    fig.savefig(output_dir / figurename, format="png", bbox_inches='tight')


def plot_power_curve(file_path, ref_turbine='NREL 5 MW'):

    """
    Plot the power curve of a wind turbine based on
    wind speed and power output data.

    Parameters:
        file_path (Path): Path to the CSV file containing
        wind speed and power data.
        ref_turbine (str): string containing the name of the
        turbine for which power data is being plotted.

    Returns:
        None: Saves the plot as a PNG file in the output directory.
    """
    # Load the data from the specified file
    df = load_data(file_path)

    # Create a figure and axis for the plot
    fig, axs = plt.subplots(figsize=(6, 6))

    # Plot the power curve (Power [kW] vs. Wind Speed [m/s])
    axs.plot(df['Wind Speed [m/s]'], df['Power [kW]'])

    # Set the title and labels for the plot
    axs.set_title(f"Power Curve, ({ref_turbine})")
    axs.set_xlabel('Wind Speed [m/s]')
    axs.set_ylabel('Power [kW]')

    # Add a grid to the plot for better readability
    axs.grid(True)

    # Adjust the layout to prevent overlapping of elements
    fig.tight_layout()

    # Define the output directory for saving the plot
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "data_files_you_generate"

    # Generate a file name for the power curve plot
    figurename = "power_curve.png"

    # Save the plot as a PNG file in the specified output directory
    fig.savefig(output_dir / figurename, format="png", bbox_inches='tight')

    # Return nothing as the function is for plotting and saving
    return


def calculate_bin_probabilities(data, bins):
    """
    Calculate the probabilities of data falling within specified bins.

    Parameters:
        data (pd.Series): The wind speed data.
        bins (list): The edges of the bins (e.g., [0, 5, 10, 15, 20]).

    Returns:
        dict: A dictionary where keys are bin ranges
        and values are probabilities (percentages).
    """
    # Use pandas cut to categorize data into bins
    bin_counts = pd.cut(data, bins=bins, right=False).value_counts(sort=False)

    # Calculate total data points
    total_count = len(data)

    # Calculate probabilities (percentages) for each bin
    bin_probabilities = {f"[{interval.left}, {interval.right})":
                         (count / total_count) * 100
                         for interval, count in bin_counts.items()}

    return bin_probabilities


def calculate_aep(bin_probabilities, power_per_bin):
    """
    Calculate the Annual Energy Production (AEP).

    Parameters:
        bin_probabilities (dict): A dictionary where keys are bin ranges
        and values are probabilities (percentages).
        power_per_bin (dict): A dictionary where keys are bin ranges
        and values are power values (e.g., kW).

    Returns:
        float: The calculated AEP.
    """
    # Ensure the bin ranges in both dictionaries match
    if set(bin_probabilities.keys()) != set(power_per_bin.keys()):
        raise ValueError("Bin ranges in probabilities "
                         "and power values do not match.")

    # Calculate AEP using the formula
    aep = 8760 * sum((bin_probabilities[bin_range] / 100)
                     * power_per_bin[bin_range]
                     for bin_range in bin_probabilities)

    return aep


def dominant_wind_direction(direction_series, bin_size=30):
    """
    Identifies the dominant wind direction range.

    Parameters:
        direction_series (pd.Series): Wind direction data in degrees [0, 360).
        bin_size (int): Size of each directional bin (default is 30°).

    Returns:
        dominant_range (str): The most frequent wind direction bin as a string.
        count (int): Number of occurrences in the dominant bin.
    """
    # Ensure valid range
    directions = direction_series % 360

    # Create bins and labels
    bins = np.arange(0, 361, bin_size)
    labels = [f"{int(bins[i])}°–{int(bins[i+1])}°" for i in range(len(bins)-1)]

    # Bin the data
    binned = pd.cut(directions,
                    bins=bins,
                    labels=labels,
                    right=False,
                    include_lowest=True)

    # Find the most frequent bin
    dominant_range = binned.value_counts().idxmax()
    count = binned.value_counts().max()

    return str(dominant_range), count


def generate_power_per_bin(nrel_data):
    """
    Generate the power per bin dictionary dynamically from the NREL data.

    Parameters:
        nrel_data (pd.DataFrame): DataFrame containing
        wind speed and power data.

    Returns:
        dict: A dictionary where keys are bin ranges
        and values are power values (kW).
    """
    power_per_bin = {}
    wind_speeds = nrel_data['Wind Speed [m/s]'].tolist()
    powers = nrel_data['Power [kW]'].tolist()

    # Create bin ranges and map them to power values
    for i in range(len(wind_speeds) - 1):
        bin_range = f"[{wind_speeds[i]:.1f}, {wind_speeds[i+1]:.1f})"
        power_per_bin[bin_range] = powers[i]

    return power_per_bin
