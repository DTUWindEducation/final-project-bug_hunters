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