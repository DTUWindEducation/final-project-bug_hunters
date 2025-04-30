from pathlib import Path
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import WRA_Package as wra

FILE_PATH = Path(__file__)
FILE_DIR = FILE_PATH.parent.parent
DATA_DIR= FILE_DIR / 'inputs'


def test_load_wind_data():
    """Test loading wind data from NetCDF file."""
    # Given
    path_nc = DATA_DIR / "1997-1999.nc"  # path to NetCDF file

    # When

    wind_speed, wind_dir, time, lat, lon = wra.load_data(path_nc, level=100)

    # Then
    assert isinstance(wind_speed, xr.DataArray)     #"Output should be an xarray DataArray"
    assert wind_speed.ndim == 3     #"Expected wind speed to have 3 dimensions (time, lat, lon)"  #array has 3 dimesnions: time, latitude, longitude


def test_plot_wind_speed_histogram(monkeypatch):  # use a pytest "monkeypatch" to stop plots from popping up
    """Check some aspects of plot_resp"""
    monkeypatch.setattr(plt, 'show', lambda: None)  # temporarily overwrite plt.show() to do nothing
    # given
    path_wind_data = DATA_DIR / "1997-1999.nc"
    wind_speeds, wind_dir, time, lat, lon = wra.load_data(path_wind_data,100)
    # when
    fig, axs = wra.plot_wind_speed_histogram(wind_speeds, 100)
    # then
    assert isinstance(fig, plt.Figure)  # check it's a figure

def test_plot_wind_time_series(monkeypatch):  # use a pytest "monkeypatch" to stop plots from popping up
    """Check some aspects of plot_resp"""
    monkeypatch.setattr(plt, 'show', lambda: None)  # temporarily overwrite plt.show() to do nothing
    # given
    path_wind_data = DATA_DIR / "1997-1999.nc"
    wind_speeds, wind_dir, time, lat, lon = wra.load_data(path_wind_data,100)
    # when
    fig, axs = wra.plot_wind_time_series(wind_speeds, time, 100)
    # then
    assert isinstance(fig, plt.Figure)  # check it's a figure

def test_calculate_alpha_dynamic_known_values():
    u_10 = np.array([5.0])
    u_100 = np.array([10.0])

    expected_alpha = np.log(10.0 / 5.0) / np.log(10)

    alpha = wra.calculate_alpha_dynamic(u_10, u_100)

    assert np.allclose(alpha, expected_alpha), f"Expected {expected_alpha}, got {alpha}"

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