# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:49:40 2021

This is the main code to generate STREAM ensembles.

Input: 
IMERG precipitation: regridding hourly 0.1deg IMERGE.hourly.nc; 
MERRA2 wind U,V components : regridding hourly 0.1deg MERRA2_0.1.hourly.nc; 
IMERG has its own motion vector, which is 0.1deg half-hourly. If available, it can be used as well!
Here we use MERRA U850,V850 as an example.
CSGD error model: csgd_NLmodel_WAR.nc. CSGD model could use GPM-2BCMB as reference!
The validation work about this choice is in
Li, Z., Wright, D. B., Hartke, S. H., Kirschbaum, D. B., Khan, S., 
Maggioni, V., & Kirstetter, P.-E. (2023). 
Toward a Globally-Applicable Uncertainty Quantification Framework 
for Satellite Multisensor Precipitation Products Based on GPM DPR. 
IEEE Transactions on Geoscience and Remote Sensing, 61, 1-15. 
doi:10.1109/tgrs.2023.3235270
details about CSGD is on https://github.com/KaidiWisc/CSGD_error_model.git

Output:
noise netcdf file
STREAM-Sat precipitation netcdf file

Reference:
Hartke, S. H., Wright, D. B., Li, Z., Maggioni, V., Kirschbaum, D. B., & Khan, S. 
(2022). Ensemble Representation of Satellite Precipitation Uncertainty 
Using a Nonstationary, Anisotropic Autocorrelation Model. 
Water Resources Research, 58(8). doi:10.1029/2021wr031650

After generating the ensembles, you could evaluate them using CRPS_cal.py and CR_cal.py

@author: Kaidi Peng
"""

from datetime import date, timedelta, datetime
from STREAM_PrecipSimulation import simulatePrecip
from STREAM_NoiseGeneration import generateNoise
from netCDF4 import Dataset,date2num
import numpy as np
import time

global startTime
startTime = time.time()

# -----------------------------------------------------------------------------
# -------------------  INPUT PARAMETERS SECTION  ------------------------------


nEns = 2 # number of ensemble members to generate
# each ensemble is independent
# parallel run for each ensemble is recommended. 
# python multiprocessing package or high throughput computation system could be used.

dt = date(2017,8,1) # date to start simulation at 2017.8.1 to 2017.8.31 

ts = 31*24 # number of timesteps to run simulation for [hrs]

obsInFname = "IMERGE.hourly.nc"  # 2017 whole year

windInFname ="MERRA2_0.1.hourly.nc"  # Whole year

paramsInFname = "csgd_NLmodel_WAR.nc"  # 

CSGDWin=10  #CSGD model training window (i.e., the resolution of CSGD models, 10 in this example)


# ---  output file names  ---

end_dt = dt + timedelta(hours=(ts-1)) # end date of simulation
noiseOutFname =  "noise_%s_%s.nc"%(dt.strftime('%Y%m%d'),end_dt.strftime('%Y%m%d'))  # path for noise ensemble output

precipOutFname = "STREAM_%s_%s.nc"%(dt.strftime('%Y%m%d'),end_dt.strftime('%Y%m%d'))  # path for precipitation ensemble output


# ---------------  END OF INPUT PARAMETERS SECTION FOR STREAM  ----------------
# -----------------------------------------------------------------------------
#%%

# --- Generate noise ensemble and save to netcdf noiseOutFname
generateNoise(nEns,ts,dt,obsInFname,windInFname,noiseOutFname)
executionTime = (time.time() - startTime)
print('Noise finished. time in minutes: ' + str(executionTime/60))

# --- Simulate STREAM ensemble of precipitation
simPrcp = simulatePrecip(dt,nEns,ts,obsInFname,noiseOutFname,paramsInFname,CSGDWin)

# ----- Save simulated ensemble of precip to netcdf precipOutFname ------------

ysize = np.shape(simPrcp)[2]
xsize = np.shape(simPrcp)[3]

new_cdf = Dataset(precipOutFname, 'w', format = "NETCDF4", clobber=True)

# create array of time stamps
time_hrs = [datetime(dt.year,dt.month,dt.day,0,0)+n*timedelta(hours=1) for n in range(ts)]
units = 'hours since 1970-01-01 00:00:00 UTC'

# create dimensions
new_cdf.createDimension('lat', ysize)
new_cdf.createDimension('lon', xsize)
new_cdf.createDimension('ens_n', nEns)
new_cdf.createDimension('time', ts)

# write time stamps to variable
timevar = new_cdf.createVariable('time','d', ('time'))
timevar.units = units
timevar[:] = date2num(time_hrs,units,calendar="gregorian")

# add lat and lon variables
latitude = new_cdf.createVariable('latitude', 'f4', ('lat'), zlib=True,least_significant_digit=2)
latitude.units = 'degrees_north'
latitude.long_name = 'latitude'
# open input precipitation netcdf to get latitude and longitude arrays
ds = Dataset(obsInFname)
latitude[:] = ds['latitude'][:]

longitude = new_cdf.createVariable('longitude', 'f4', ('lon'), zlib=True,least_significant_digit=2)
longitude.units = 'degrees_east'
longitude.long_name = 'longitude'
longitude[:] = ds['longitude'][:]

ds = None

prcp = new_cdf.createVariable('prcp', 'f4', ('lat','lon','time','ens_n'), zlib=True,least_significant_digit=3)
prcp.units = 'mm/hr'
prcp.long_name = 'Simulated Precipitation'
prcp[:,:,:,:] = np.transpose(simPrcp,[2,3,1,0])  # ens-time-lat-lon -> lat-lon-time-ens

new_cdf.close()



# --------- ----- END OF STREAM SIMULATION CODE  ------------------------------
# -----------------------------------------------------------------------------
executionTime = (time.time() - startTime)
print('Total execution time in minutes: ' + str(executionTime/60))
