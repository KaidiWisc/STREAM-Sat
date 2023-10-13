# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 22:18:25 2022

@author: Kaidi Peng

WAR: Wetted Area Ratio
the percentage of pixels with positive precipitation in each box centered on each pixel.
Range: 0-1
The default window size is 21*21, which means the radius of the moving window is 10.

Input: IMERG netcdf file
Ourput: Corresponding WAR netcdf file

WAR is used as covariate in CSGD model.
It is also needed in STREAM-Sat ensemble generation.
"""

import numpy as np
from netCDF4 import Dataset,date2num
from datetime import datetime, timedelta
import xarray as xr

area_set = False
#==========================================================================
def set_area(ysize,xsize,dset=21):
    # the default window size is 21*21
    global area
    area = np.zeros((ysize,xsize))
    d = dset
    area.fill((d)**2)
    rads=(d-1)/2 # if d=21, rads=10
    a = np.reshape(np.arange((rads+1),d),(1,rads))
    b = np.reshape(np.arange((d-1),rads,-1),(1,rads))
    
    area[:rads,:rads] = np.transpose(a)*a
    area[(ysize-rads):,:rads] = np.transpose(b)*a
    area[:rads,(xsize-rads):] = np.transpose(a)*b
    area[(ysize-rads):,(xsize-rads):] = np.transpose(b)*b
    
    for i in range(rads+1,d):
        area[i-(rads+1),rads:(xsize-rads)] = i*d
        area[(ysize+rads)-i,rads:(xsize-rads)] = i*d
        area[rads:(ysize-rads),i-(rads+1)] = i*d
        area[rads:(ysize-rads),(xsize+rads)-i] = i*d
    
    global area_set
    area_set = True
    
def getWARfield(field,r=10):
    
    ysize = np.shape(field)[0]
    xsize = np.shape(field)[1]
    
    if area_set == False:
        set_area(ysize,xsize)
    
    rainy = np.zeros(np.shape(field))
    # default rain threshold: 0.1mm/hr
    # when the observed precip. > 0.1mm/hr, it is considered as rain
    rainy[field>=0.1] = 1 

    rainysum = np.zeros(np.shape(field))
    

    for i in range(-r,r+1):
        
        irange = (np.max((0,-i)),np.min((xsize,xsize-i)))
        
        for j in range(-r,r+1):
            
            subfield = rainy[np.max((0,j)):np.min((ysize,ysize+j)),np.max((0,i)):np.min((xsize,xsize+i))]
            
            jrange = (np.max((0,-j)),np.min((ysize,ysize-j)))
            
            rainysum[jrange[0]:jrange[1],irange[0]:irange[1]] += subfield
    
    
    WARfield = rainysum/area
    
    return(WARfield)

def writeWARfile(imerg,latstart,latend,lonstart,lonend,d=21):
    
    ts = np.shape(imerg)[2]
    ysize = np.shape(imerg)[0]
    xsize = np.shape(imerg)[1]
    WAR = np.zeros((ysize,xsize,ts),dtype=np.float32)

    fname="WAR.r%d.hourly.nc"%(int((d-1)/2))

    print('Writing %s \n'%fname)


    # create array of time stamps
    time_hrs = [datetime(2017,1,1,0,0)+n*timedelta(hours=1) for n in range(ts)]
    units = 'hours since 1970-01-01 00:00:00 UTC'

    new_file = fname

    new_cdf = Dataset(new_file, 'w', format = "NETCDF4", clobber=True)

    # create dimensions
    new_cdf.createDimension('lat', ysize)
    new_cdf.createDimension('lon', xsize)
    new_cdf.createDimension('time', ts)

    # write time stamps to variable
    time = new_cdf.createVariable('time','d', ('time'))
    time.units = units
    time[:] = date2num(time_hrs,units,calendar="gregorian")

    # add lat, and lon variables
    latitude = new_cdf.createVariable('latitude', 'f8', ('lat'), zlib=True)
    latitude.units = 'degrees_north'
    latitude.long_name = 'latitude'
    latitude[:] = np.arange(latstart-0.05,latend,-0.1)
    
    longitude = new_cdf.createVariable('longitude', 'f8', ('lon'), zlib=True)
    longitude.units = 'degrees_east'
    longitude.long_name = 'longitude'
    longitude[:] =  np.arange(lonstart+0.05,lonend,0.1)

    prcp = new_cdf.createVariable('war', 'f8', ('lat','lon','time'), zlib=True, least_significant_digit=4)
    prcp.units = '--'
    prcp.long_name = 'Wetted Area Ratio'

    for t in range(0,ts):      
        WAR[:,:,t] = getWARfield(imerg[:,:,t], 10)

    del imerg
    prcp[:,:,:] = WAR

    new_cdf.close()
#============================================================================================
# 

dsIMERG = xr.open_dataset("IMERGE.hourly.nc")

imerg = dsIMERG['prcp'].data
lats0=dsIMERG["latitude"].data[0]
lats1=dsIMERG["latitude"].data[-1]
lons0=dsIMERG["longitude"].data[0]
lons1=dsIMERG["longitude"].data[-1]

# default rain threshold: 0.1mm/hr
# when the observed precip. > 0.1mm/hr, it is considered as rain
imerg[imerg<0.1]=0
writeWARfile(imerg,lats0,lats1,lons0,lons1)
