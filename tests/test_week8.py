from pathlib import Path
import numpy as np
import xarray as xr
import WRA_Package as wra

FILE_PATH = Path(__file__)
FILE_DIR = FILE_PATH.parent.parent
DATA_DIR= FILE_DIR / 'inputs'

def test_load_wind_data():
    """Test loading wind data from NetCDF file."""
    # Given
    path_nc = DATA_DIR / "1997-1999.nc"  # path to NetCDF file

    # When
    wind_speed = wra.load_data(path_nc, level="100m")

    # Then
    assert isinstance(wind_speed, xr.DataArray), "Output should be an xarray DataArray"
    assert wind_speed.ndim == 3, "Expected wind speed to have 3 dimensions (time, lat, lon)"  #array has 3 dimesnions: time, latitude, longitude