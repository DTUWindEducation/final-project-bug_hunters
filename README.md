[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)
# Our Great Package

Team: bug_hunters

## Overview

This project centres on building a Python package that can be used to take provided ERA5 reanalysis data and processes it into practical wind‑resource insights for a chosen site. Using several years of hourly wind speed data collected at 10 m and 100 m heights as well as referance wind turbine data, the package can be used to process the data, producing the following results:

        - Load and parse the data from an .nc file
        - Generate wind speed and wind direction time series at 10 m and 100 m height at each of the four provided locations as well as for a chosen location within the specified grid. 
        - Compute a wind speed time series for a selected height for each of the gour provided locations. 
        - Fit a Weibull distribution for wind speed for a chosen location within the specified grid and at a selected height. 
        - Plot a wind rose diagram for a specified location within the grid and height.
        - Compute the AEP of the NREL 5 MW turbine at a given location and height for a given year. 
        - Plot the power curve for the NREL 5 MW turbine. 
        - Compute the dominant wind direction for the time span of the provided data sets. 

The project was developed in two main parts: the creation of the package (WRA_package) where the functions for this package are contained in the __init__.py file, and the creation of the main.py script which illustrated how the package functions can be implemented and saves relevant data and figures produced by the script. In addition to these parts, tests were created to verify that the package functions behaved as expected, producing the expected results. The final result produces a package which can be utilized to conduct a wind resource assessment using reanalysis data. The package can be used to produce any of the previously described results for such an assessment. 


## Quick-start guide

1. Clone the repository
2. Note that the folder structure must not be changed after cloning the repository in order for the script in main.py to run.
3. Open an anaconda prompt and cd into the cloned bug_hunters repo. Use the anaconda promtp to complete steps 4 - 8. 
4. Create a new environment using the following command: 
        - conda create -n <your environment name> python=3.11 -y 
5. Activate the environment using the following command: 
        - conda activate <your environment name> 
6. Install the necessary packages required for the WRA_Package. The necessary packages are listed below: 
        - pandas 
        - matplot lib 
        - xarray 
        - netCDF4
        - pathlib 
        - scipy 
        - pytest 
        - windrose 
        - DOUBLE CHECK IF THERE ARE ANY OTHER REQUIRED PACKAGES
7. Activate the environment using the following command: 
        - conda activate <your environment name>
8. Install the wind resource assessment package (WRA_Package) using the following command: 
        - pip install -e .
9. Ensure that the following data files are present in the 'inputs' folder: 
        - 1997-199.nc 
        - 2000-2002.nc
        - 2003-2005.nc
        - 2006-2008.nc 
        - NREL_Reference_5MW_126.csv
        - NREL_Reference_15MW_240.csv
10. Run main.py and the script will processes the multi-year hourly wind data and one of the data files containing the referance turbine data, producing the results described in the 'Overview' section
11. The data and figures produced by the script will save locally in the cloned repo in the 'outputs' folder. 


## Architecture

ADD: description of architecture and class 

## Peer review

Each team member was responsible for specific tasks, as shown below. Members reviewed eachothers work using pull requests. One member would create a pull request for their work, and a different members would review the work and accept the request if the work appeared sufficient. Feedback was provided for areas of each group members codes based on this pull request methodology. If an area of the code appeared incorrect or could be improved, this feedback was relayed to the group member responsible for that portion of the project. The group was frequently in communication using messages, video calls, or in-person meetins. Video calls and in-person meetings were often used to collaborate in order to solve issues which group members were encountering. These meetings were used as a time to brianstorm solutions to errors, collaborate on code, and assist in debugging. 

Task delegation:
Kali: Task 1,2,3, pylint score checking, README file,
Tessa: Task 4,5,6, Collaboration.md, Architecture diagram
Benni: Task 7,8, Collaboration.md, Docstrings, Comments
