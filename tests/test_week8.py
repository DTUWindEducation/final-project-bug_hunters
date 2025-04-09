from pathlib import Path
import numpy as np
import xarray as xr
import sys
sys.path.append(str(Path(__file__).resolve().parents[1])) #goes two levels up to find src folder

from src import load_wind_data

DATA_DIR= Path('./inputs')

def test_load_wind_data():
    """Test loading wind data from NetCDF file."""
    # Given
    path_nc = DATA_DIR / "1997-1999.nc"  # path to NetCDF file

    # When
    wind_speed = load_wind_data(path_nc, level="100m")

    # Then
    assert isinstance(wind_speed, xr.DataArray), "Output should be an xarray DataArray"
    assert wind_speed.ndim == 3, "Expected wind speed to have 3 dimensions (time, lat, lon)"  #array has 3 dimesnions: time, latitude, longitude