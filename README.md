# Wind Speed Recovery Promotes Vegetation Growth

codes for "Wind Speed Recovery Promotes Vegetation Growth"

## Contents
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Package Tests](#package-tests)
- [Demo](#demo)
- [Reproduction](#reproduction)
- [Data Verification](#data-verification)


# System Requirements
## Hardware Requirements
The environment requires only a standard computer with enough RAM to support the operations defined by a user. We use a computer with the following specs:

RAM: 16+ GB  
CPU: Intel® Core™ i7-10750H @ 2.60GHz, 6 cores / 12 threads

## Software Requirements

### OS Requirements

This project was developed using:

-Python 3.11.9 (via conda-forge)

-IPython 8.24.0

-OS: Windows 10/11 64-bit  |  Version 22H2 

-Architecture: AMD64

The codes should be compatible with Windows, Mac, and Linux operating systems.

# Installation Guide
First, download miniconda

```
https://www.anaconda.com/download
```
which should install in a few minutes.
After finish downloading, open Anaconda Powershell Prompt or  Anaconda Prompt.
Then create a new environment:
```
conda create -n myenv python=3.11
```
We use python version = 3.11.9, when create the new environment, the version should be more than 3.9.
You can use whatever Python IDE you like, here we use spyder:
```
conda activate myenv
```
```
conda install spyder
```
```
spyder
```
After the order "spyder" in conda, spyder will be open.

### Package Installation
Then make sure the following Python libraries are installed: numpy, pandas, xarray, matplotlib, pwlf, cartopy，rpy2, geopandas, salem, R package relaimpo. You can use the following order in your conda environment:
```
pip install numpy pandas xarray matplotlib pwlf cartopy geopandas salem rioxarray pytz
```
pip install packages will cost about few minutes.


# Package tests
In spyder, run the following orders:
```
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pwlf
import cartopy.crs as ccrs
import geopandas as gpd
import salem

```
if successfully run, then the packages are successfully downloaded.

# Demo
1. Finish Installation Guide and  Package tests
2. Download Things in demo folder
3. change the path in the code to the path after downloading
4. Run demo.py

The whole demo will be finished in few minutes.

The Demo has demonstrated the usage of the relevant libraries and production of results. 

# Reproduction

Run Fig1~Fig4.py cells one by one, and the path should be replaced by requirements.

The LAI data can download from: https://doi.org/10.5281/zenodo.7649107

The wind data can download from: https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/

The land cover data can be downloaded from https://cds.climate.copernicus.eu/datasets/satellite-land-cover?tab=overview, and the filename is "ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7cds.nc" and "C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc"

The "LAIdataproduce.py" and "WINDdataproduce.py"  contain the preprocessing of the raw data, while "Fig1~Fig4.py" include data processing and visualization for the figures in the paper.

PS: the path in code: ('F:/merged_filtered_common_stations.csv') should be replaced by the data after running WINDdataproduce.py
the path in code: ('F:/LAI.csv') should be replaced by the data after running LAIdataproduce.py

LAI data used in the code should run LAIdataproduce.py first.
wind data used in the code should run WINDdataproduce.py first.


After completing the previous steps, Figures 1 to figures 4 should be visualized in Spyder within a minute.


# Data Verification
You can verification the LAI dealing process by visualization of 'G:/CN05.1/month/0.25_month_onlychina.nc' in LAI data process.






