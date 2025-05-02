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
    """Check some aspects of plot_resp"""
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
    Check the results of the wind interpolation method

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
    u_10 = np.array([5.0])
    u_100 = np.array([10.0])

    expected_alpha = np.log(10.0 / 5.0) / np.log(10)

    alpha = wra.calculate_alpha_dynamic(u_10, u_100)

    assert np.allclose(alpha, expected_alpha), f"Expected {expected_alpha}, got {alpha}"

def test_calculate_alpha_dynamic_zero_division():
    u_10 = np.array([0.0, 5.0])
    u_100 = np.array([5.0, 10.0])

    alpha = wra.calculate_alpha_dynamic(u_10, u_100)

    assert np.isfinite(alpha).all(), "Alpha should not contain NaNs or Infs"
    assert (alpha >= 0).all(), "Alpha should be >= 0"

def test_extrapolate_wind_speed_known_values():
    """
    Test extrapolated wind speed with known dynamic alpha.
    """
    u_10 = np.array([5.0])
    u_100 = np.array([10.0])
    u_ref = np.array([5.0])  # same as u_10
    z_ref = 10
    z_target = 50

    expected_alpha = np.log(10.0 / 5.0) / np.log(10)
    expected_speed = u_ref * (z_target / z_ref) ** expected_alpha

    result = wra.extrapolate_wind_speed(u_ref, u_10, u_100, z_ref, z_target)

    assert np.allclose(result, expected_speed), f"Expected {expected_speed}, got {result}"

def test_extrapolate_wind_speed_array_input():
    u_ref = np.array([5.0, 6.0, 7.0])
    u_10 = np.array([5.0, 6.0, 7.0])
    u_100 = np.array([10.0, 11.0, 12.0])
    z_ref = 10
    z_target = 100

    result = wra.extrapolate_wind_speed(u_ref, u_10, u_100, z_ref, z_target)
    
    assert result.shape == u_ref.shape, "Output shape mismatch"
    assert (result > 0).all(), "Extrapolated speeds should be positive"

def test_fit_weibull_distribution():
    """
    Test that Weibull fitting returns plausible shape and scale.
    """
    np.random.seed(42)
    sample_data = np.random.weibull(a=2.0, size=1000) * 8  # simulate wind speeds

    shape, scale = wra.fit_weibull_distribution(sample_data)

    assert shape > 0, "Shape parameter should be > 0"
    assert scale > 0, "Scale parameter should be > 0"

def test_plot_wind_speed_with_weibull(monkeypatch):
    """
    Test wind speed + Weibull plot returns a valid figure.
    """
    monkeypatch.setattr(plt, 'show', lambda: None)

    np.random.seed(1)
    wind_speeds = np.random.weibull(a=2.0, size=1000) * 7
    shape, scale = wra.fit_weibull_distribution(wind_speeds)

    fig, ax = wra.plot_wind_speed_with_weibull(wind_speeds, shape, scale)

    assert isinstance(fig, plt.Figure), "Expected a matplotlib Figure"
    assert hasattr(ax, "plot"), "Expected an Axes object"

def test_plot_wind_rose():
    """
    testing the wind rose plotting function
    """

    # given 
    path_wind_data = DATA_DIR / "1997-1999.nc"
    lat = 55.5 
    lon = 7.75
    height = 10
    df = wra.load_data(path_wind_data)
    df_2 = wra.compute_and_plot_time_series(df,lat,lon,height, display_figure=False)
    direction = df_2['direction']            # degrees
    speed     = df_2['speed']            # m/s
    num_bins  = 4

    # for
    wra.plot_wind_rose(direction, speed, num_bins=num_bins, label_interval=45)

    ax = plt.gca()
    assert isinstance(ax, wra.WindroseAxes)        

def test_calculate_bin_probabilities():
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

def test_calculate_aep():
    # Sample bin probabilities and power per bin
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
 