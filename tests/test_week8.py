from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib 
matplotlib.use("Agg")       # included to resolve tk issue (recommended to include at advice of teaching staff)
import matplotlib.pyplot as plt
import WRA_Package as wra
from WRA_Package import WindInterpolator
from WRA_Package import separate_data_by_year
from WRA_Package import generate_power_per_bin
import pytest

FILE_PATH = Path(__file__)
FILE_DIR = FILE_PATH.parent.parent
DATA_DIR= FILE_DIR / 'inputs'


def test_load_wind_data_nc():
    """Test loading wind data from NetCDF file."""
    # Given
    path_nc = DATA_DIR / "1997-1999.nc"  # path to NetCDF file

    # When

    df = wra.load_data(path_nc)

    # Then
    assert isinstance(df, pd.DataFrame)   

def test_load_wind_data_csv():
    """Test loading wind data from NetCDF file."""
    # Given
    path_csv = DATA_DIR / "NREL_Reference_15MW_240.csv"  # path to NetCDF file

    # When

    df = wra.load_data(path_csv)

    # Then
    assert isinstance(df, pd.DataFrame)   


def test_conc_data(): 

    """
    Test the `conc_data` function for concatenating wind data from multiple files.

    This test ensures that the function correctly combines data from multiple NetCDF files
    into a single DataFrame.
    """
    # given 
    FILE_PATH = Path(__file__)      # path to this file
    FILE_DIR = FILE_PATH.parent.parent     # path to main folder 
    DATA_DIR = FILE_DIR / 'inputs'
    DATA_97_99 = DATA_DIR / '1997-1999.nc'
    DATA_00_02 = DATA_DIR / '2000-2002.nc'
    DATA_03_05 = DATA_DIR / '2003-2005.nc'
    DATA_06_08 = DATA_DIR / '2006-2008.nc'
    data_list = [DATA_97_99,DATA_00_02, DATA_03_05, DATA_06_08]  

    # when 
    df = wra.conc_data(data_list)

    # then 
    assert isinstance(df, pd.DataFrame)

def test_separate_data_by_year():

    """
    Test the `separate_data_by_year` function.

    This test ensures that the function correctly filters wind data for a specific year
    and raises an appropriate error when no data is available for the requested year.

    Steps:
    1. Create a sample DataFrame with wind data for multiple years.
    2. Call the `separate_data_by_year` function to filter data for a specific year.
    3. Assert that the filtered data contains only rows for the requested year.
    4. Assert that the function raises a ValueError when no data is available for the requested year.
    """
     
    # Sample wind data
    data = {
        "valid_time": [
            "2005-01-01 00:00:00", "2005-02-01 00:00:00", 
            "2006-01-01 00:00:00", "2006-02-01 00:00:00"
        ],
        "wind_speed": [5.0, 6.0, 7.0, 8.0]
    }
    wind_data = pd.DataFrame(data)

    # Test case 1: Filter data for 2005
    filtered_data = separate_data_by_year(wind_data, 2005)
    assert len(filtered_data) == 2, "Expected 2 rows for the year 2005"
    assert all(filtered_data['valid_time'].dt.year == 2005), "Filtered data contains incorrect years"

    # Test case 2: Filter data for 2006
    filtered_data = separate_data_by_year(wind_data, 2006)
    assert len(filtered_data) == 2, "Expected 2 rows for the year 2006"
    assert all(filtered_data['valid_time'].dt.year == 2006), "Filtered data contains incorrect years"

    # Test case 3: No data for the year 2007
    with pytest.raises(ValueError, match="No data found for the year 2007."):
        separate_data_by_year(wind_data, 2007)


def test_plot_wind_time_series(monkeypatch):  # use a pytest "monkeypatch" to stop plots from popping up
    """
    Test the `plot_wind_time_series` function.

    This test ensures that the function generates a valid matplotlib figure and axes
    when provided with wind data for a specific location and height.
    """
    monkeypatch.setattr(plt, 'show', lambda: None)  # temporarily overwrite plt.show() to do nothing
    # given
    path_wind_data = DATA_DIR / "1997-1999.nc"
    lat = 55.5 
    lon = 7.75
    height = 10
    df = wra.load_data(path_wind_data)
    df_2 = wra.compute_and_plot_time_series(df,lat,lon,height, display_figure=False)
    # when
    fig, axs = wra.plot_wind_time_series(df_2, lat, lon, 100)
    # then
    assert isinstance(fig, plt.Figure)  # check it's a figure
    assert axs.shape == (2,)
    assert all(isinstance(ax, plt.Axes) for ax in axs)
    assert len(fig.get_axes()) == 2          # and that there are 2 axes (since two subplots)


def test_wind_interpolation():
    """
    Test the `WindInterpolator` class for interpolating wind data.

    This test ensures that the interpolation method correctly computes wind components
    (u10, v10, u100, v100) and time values for a given location within the grid.

    """
    # given 
    path_wind_data = DATA_DIR / "1997-1999.nc"
    lat = 55.5 
    lon = 7.75
    height = 10
    df = wra.load_data(path_wind_data)
    df_2 = wra.compute_and_plot_time_series(df,lat,lon,height, display_figure=False)
    grid_points = [(55.5, 7.75), (55.5, 8.), (55.75, 7.75), (55.75, 8.)]
    time = df_2['time']

    # when 
    interp = WindInterpolator(grid_points, df)
    u10, v10, u100, v100, out_times = interp.interpolate(55.6, 7.8)

    # then
    assert isinstance(u10, np.ndarray)
    assert isinstance(v10, np.ndarray)
    assert isinstance(u100, np.ndarray) 
    assert isinstance(v100, np.ndarray)
    assert u10.shape == v10.shape == u100.shape == v100.shape == (len(time),)
    assert np.array_equal(out_times, time.values)

def test_compute_and_plot_timeseries_gridpoint(monkeypatch):
    """ 
    testing comupte and plot time series to ensure 
    that it returns a df and fix, axs for a grid point

    """
    monkeypatch.setattr(plt, 'show', lambda: None)

    # given 
    path_wind_data = DATA_DIR / "1997-1999.nc"
    lat = 55.5 
    lon = 7.75
    height = 10
    df = wra.load_data(path_wind_data)
  
    # when 
    df_2, fig, axs = wra.compute_and_plot_time_series(
        df, lat, lon, height, display_figure=True
    )

    # then
    assert isinstance(df_2, pd.DataFrame)
    assert {"time", "speed", "direction", "u10", "v10","u100","v100"}.issubset(df_2.columns)
    assert len(df_2['speed']) == (len(df['u10'])/4)     # dividing by 4 since there is data for 4 different locations
    assert isinstance(fig, plt.Figure)
    assert isinstance(axs, np.ndarray) and axs.shape == (2,)
    assert all(isinstance(ax, plt.Axes) for ax in axs)
    assert len(fig.get_axes()) == 2

def test_compute_and_plot_timeseries_interpolation(monkeypatch):
    """
    testing comupte and plot time series to ensure 
    that it returns a df and fix, axs for a point 
    within the grid 

    """
    monkeypatch.setattr(plt, 'show', lambda: None)

    # given 
    path_wind_data = DATA_DIR / "1997-1999.nc"
    lat = 55.63 
    lon = 7.82
    height = 10
    df = wra.load_data(path_wind_data)
  
    # when 
    df_2, fig, axs = wra.compute_and_plot_time_series(
        df, lat, lon, height, display_figure=True
    )

    # then
    assert {"time", "speed", "direction", "u10", "v10","u100","v100"}.issubset(df_2.columns)
    assert len(df_2['speed']) == (len(df['u10'])/4)     # dividing by 4 since there is data for 4 different locations
    assert isinstance(fig, plt.Figure)
    assert isinstance(axs, np.ndarray) and axs.shape == (2,)
    assert all(isinstance(ax, plt.Axes) for ax in axs)
    assert len(fig.get_axes()) == 2

def test_calculate_alpha_dynamic_known_values():
    """
    Test the `calculate_alpha_dynamic` function with known values.

    This test verifies that the function correctly calculates the wind shear exponent (alpha)
    using the logarithmic wind profile formula for given wind speeds at 10m and 100m heights.

    Steps:
    1. Define known wind speeds at 10m and 100m heights.
    2. Calculate the expected alpha value using the formula.
    3. Call the `calculate_alpha_dynamic` function to compute alpha.
    4. Assert that the computed alpha matches the expected value.

    """
    # Given: Wind speeds at 10m and 100m heights
    u_10 = np.array([5.0])
    u_100 = np.array([10.0])

    # Expected alpha value calculated using the logarithmic wind profile formula
    expected_alpha = np.log(10.0 / 5.0) / np.log(10)

    # When: Call the function to calculate alpha
    alpha = wra.calculate_alpha_dynamic(u_10, u_100)

    # Then: Assert that the calculated alpha matches the expected value
    assert np.allclose(alpha, expected_alpha), f"Expected {expected_alpha}, got {alpha}"

def test_calculate_alpha_dynamic_zero_division():

    """
    Test the `calculate_alpha_dynamic` function for edge cases involving zero wind speeds.

    This test ensures that the function handles cases where wind speeds at 10m or 100m are zero
    without producing NaN or infinite values in the calculated alpha.

    Steps:
    1. Define wind speeds at 10m and 100m, including a zero value.
    2. Call the `calculate_alpha_dynamic` function to compute alpha.
    3. Assert that the computed alpha does not contain NaN or infinite values.
    4. Assert that all alpha values are non-negative.
    """

    # Given: Wind speeds at 10m and 100m, including a zero value
    u_10 = np.array([0.0, 5.0])
    u_100 = np.array([5.0, 10.0])

    # When: Call the function to calculate alpha
    alpha = wra.calculate_alpha_dynamic(u_10, u_100)

    # Then: Assert that the calculated alpha does not contain NaN or infinite values
    assert np.isfinite(alpha).all(), "Alpha should not contain NaNs or Infs"
    assert (alpha >= 0).all(), "Alpha should be >= 0"

def test_extrapolate_wind_speed_known_values():
    """
    Test extrapolated wind speed with known dynamic alpha.
    """

    # Define wind speeds at 10m and 100m heights
    u_10 = np.array([5.0]) # Wind speed at 10m
    u_100 = np.array([10.0]) # Wind speed at 100m
    u_ref = np.array([5.0])  # Reference wind speed (same as u_10)
    z_ref = 10 # Reference height in meters
    z_target = 50 # Target height in meters

    # Calculate the expected wind shear exponent (alpha) using the logarithmic wind profile formula
    expected_alpha = np.log(10.0 / 5.0) / np.log(10)

    # Calculate the expected wind speed at the target height using the power law formula
    expected_speed = u_ref * (z_target / z_ref) ** expected_alpha

    # Call the `extrapolate_wind_speed` function to compute the wind speed at the target height
    result = wra.extrapolate_wind_speed(u_ref, u_10, u_100, z_ref, z_target)

    # Assert that the computed wind speed matches the expected wind speed
    assert np.allclose(result, expected_speed), f"Expected {expected_speed}, got {result}"

def test_extrapolate_wind_speed_array_input():
    """
    Test the `extrapolate_wind_speed` function with array inputs.

    This test ensures that the function can handle array inputs for wind speeds
    and correctly extrapolates wind speeds to the target height.

    Steps:
    1. Define arrays for wind speeds at 10m, 100m, and reference height.
    2. Call the `extrapolate_wind_speed` function with these arrays.
    3. Assert that the output shape matches the input shape.
    4. Assert that all extrapolated wind speeds are positive.
    """
    # Given: Arrays for wind speeds at 10m, 100m, and reference height
    u_ref = np.array([5.0, 6.0, 7.0])  # Reference wind speeds
    u_10 = np.array([5.0, 6.0, 7.0])  # Wind speeds at 10m
    u_100 = np.array([10.0, 11.0, 12.0])  # Wind speeds at 100m
    z_ref = 10  # Reference height in meters
    z_target = 100  # Target height in meters

    # When: Call the function to extrapolate wind speeds
    result = wra.extrapolate_wind_speed(u_ref, u_10, u_100, z_ref, z_target)

    # Then: Assert that the output shape matches the input shape
    assert result.shape == u_ref.shape, "Output shape mismatch"

    # Assert that all extrapolated wind speeds are positive
    assert (result > 0).all(), "Extrapolated speeds should be positive"

def test_fit_weibull_distribution():
    """
    Test that Weibull fitting returns plausible shape and scale parameters.

    This test ensures that the `fit_weibull_distribution` function correctly fits
    a Weibull distribution to simulated wind speed data and returns valid shape
    and scale parameters.

    Steps:
    1. Generate simulated wind speed data using a Weibull distribution.
    2. Call the `fit_weibull_distribution` function to fit the data.
    3. Assert that the shape and scale parameters are positive.
    """
    # Generate simulated wind speed data using a Weibull distribution
    np.random.seed(42)  # Set seed for reproducibility
    sample_data = np.random.weibull(a=2.0, size=1000) * 8  # Simulate wind speeds

    # Fit the Weibull distribution to the simulated data
    shape, scale = wra.fit_weibull_distribution(sample_data)

    # Assert that the shape parameter is positive
    assert shape > 0, "Shape parameter should be > 0"

    # Assert that the scale parameter is positive
    assert scale > 0, "Scale parameter should be > 0"


def test_plot_wind_speed_with_weibull(monkeypatch):
    """
    Test that the wind speed + Weibull plot returns a valid figure.

    This test ensures that the `plot_wind_speed_with_weibull` function generates
    a valid matplotlib figure and axes when provided with wind speed data and
    fitted Weibull parameters.

    Steps:
    1. Generate simulated wind speed data using a Weibull distribution.
    2. Fit the Weibull distribution to the data.
    3. Call the `plot_wind_speed_with_weibull` function to generate the plot.
    4. Assert that the returned objects are a valid matplotlib figure and axes.
    """
    # Prevent the plot from being displayed during the test
    monkeypatch.setattr(plt, 'show', lambda: None)

    # Generate simulated wind speed data
    np.random.seed(1)  # Set seed for reproducibility
    wind_speeds = np.random.weibull(a=2.0, size=1000) * 7  # Simulate wind speeds

    # Fit the Weibull distribution to the simulated data
    shape, scale = wra.fit_weibull_distribution(wind_speeds)

    # Generate the wind speed + Weibull plot
    fig, ax = wra.plot_wind_speed_with_weibull(wind_speeds, shape, scale)

    # Assert that the returned figure is a valid matplotlib Figure object
    assert isinstance(fig, plt.Figure), "Expected a matplotlib Figure"

    # Assert that the returned axes object has a `plot` method
    assert hasattr(ax, "plot"), "Expected an Axes object"


def test_plot_wind_rose():
    """
    Test the wind rose plotting function.

    This test ensures that the `plot_wind_rose` function generates a valid wind rose
    plot using wind direction and speed data.

    Steps:
    1. Load wind data from a NetCDF file.
    2. Compute wind speed and direction time series for a specific location.
    3. Call the `plot_wind_rose` function to generate the wind rose plot.
    4. Assert that the returned axes object is a valid WindroseAxes object.
    """
    # Given: Load wind data and compute wind speed and direction
    path_wind_data = DATA_DIR / "1997-1999.nc"  # Path to the NetCDF file
    lat = 55.5  # Latitude of the location
    lon = 7.75  # Longitude of the location
    height = 10  # Height level in meters
    df = wra.load_data(path_wind_data)  # Load wind data
    df_2 = wra.compute_and_plot_time_series(df, lat, lon, height, display_figure=False)  # Compute time series
    direction = df_2['direction']  # Extract wind direction data (degrees)
    speed = df_2['speed']  # Extract wind speed data (m/s)
    num_bins = 4  # Number of bins for wind speed categories

    # When: Call the `plot_wind_rose` function
    wra.plot_wind_rose(direction, speed, num_bins=num_bins, label_interval=45)

    # Then: Assert that the returned axes object is a valid WindroseAxes object
    ax = plt.gca()  # Get the current axes
    assert isinstance(ax, wra.WindroseAxes), "Expected a WindroseAxes object"

def test_calculate_bin_probabilities():

    """
    Test the `calculate_bin_probabilities` function.

    This test ensures that the function correctly calculates the probabilities
    of wind speeds falling into specified bins.

    Steps:
    1. Define sample wind speed data and bins.
    2. Call the `calculate_bin_probabilities` function.
    3. Assert that the calculated probabilities match the expected values.
    """
    # Sample wind speed data
    data = pd.Series([3.5, 4.2, 5.1, 6.8, 7.3, 8.0, 9.5])

    # Define bins
    bins = [3, 4, 5, 6, 7, 8, 9, 10]

    # Expected probabilities (calculated manually)
    expected = {
        "[3, 4)": 14.2857,
        "[4, 5)": 14.2857,
        "[5, 6)": 14.2857,
        "[6, 7)": 14.2857,
        "[7, 8)": 14.2857,
        "[8, 9)": 14.2857,
        "[9, 10)": 14.2857,
    }

    # Call the function
    result = wra.calculate_bin_probabilities(data, bins)

    # Assert that the result matches the expected probabilities
    for bin_range, probability in expected.items():
        assert abs(result[bin_range] - probability) < 0.01

    # Assert the result matches the expected output
    assert result == expected_power_per_bin, f"Expected {expected_power_per_bin}, but got {result}"
    bin_probabilities = {
        "[3.0, 4.0)": 10.0,
        "[4.0, 5.0)": 20.0,
        "[5.0, 6.0)": 30.0,
        "[6.0, 7.0)": 40.0,
    }
    power_per_bin = {
        "[3.0, 4.0)": 100.0,
        "[4.0, 5.0)": 200.0,
        "[5.0, 6.0)": 300.0,
        "[6.0, 7.0)": 400.0,
    }

    # Expected AEP (calculated manually)
    expected_aep = 8760 * (0.1 * 100 + 0.2 * 200 + 0.3 * 300 + 0.4 * 400)

    # Call the function
    result = wra.calculate_aep(bin_probabilities, power_per_bin)

    # Assert that the result matches the expected value
    assert abs(result - expected_aep) < 1e-5

def test_generate_power_per_bin():

    """
    Test the `generate_power_per_bin` function.

    This test ensures that the function correctly generates power values for
    each wind speed bin based on the provided NREL data.

    Steps:
    1. Define sample NREL data with wind speed and power values.
    2. Call the `generate_power_per_bin` function.
    3. Assert that the generated power values match the expected output.
    """
    # Sample NREL data
    data = {
        "Wind Speed [m/s]": [3, 4, 5, 6, 7],
        "Power [kW]": [40.52, 177.67, 403.9, 737.59, 1187.18]
    }
    nrel_data = pd.DataFrame(data)

    # Expected output
    expected_power_per_bin = {
        "[3.0, 4.0)": 40.52,
        "[4.0, 5.0)": 177.67,
        "[5.0, 6.0)": 403.9,
        "[6.0, 7.0)": 737.59
    }

    # Call the function
    result = generate_power_per_bin(nrel_data)

    # Assert the result matches the expected output
    assert result == expected_power_per_bin, f"Expected {expected_power_per_bin}, but got {result}"
