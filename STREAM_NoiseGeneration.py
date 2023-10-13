# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:13:43 2020

@author: Sam Hartke

Script to generate and save netcdf of ensemble of correlated uniform noise using
pysteps replication of local spatial structures and semi-lagrangian scheme
based on MERRA2 wind fields

details about how the noise generated and pystep package is on 

[1] Pulkkinen, S., Nerini, D., Pérez Hortal, A. A., Velasco-Forero, C., Seed, A., 
Germann, U., & Foresti, L. (2019). 
Pysteps: an open-source Python library for probabilistic precipitation nowcasting (v1.0). 
Geoscientific Model Development, 12(10), 4185-4219. doi:10.5194/gmd-12-4185-2019

[2] Nerini, D., Besic, N., Sideris, I., Germann, U., & Foresti, L. (2017). 
A non-stationary stochastic ensemble generator for radar rainfall fields 
based on the short-space Fourier transform. 
Hydrol. Earth Syst. Sci., 21(6), 2777-2797. doi:10.5194/hess-21-2777-2017

"""

from netCDF4 import Dataset, date2num, num2date
import numpy as np
from datetime import date, timedelta, datetime
import math
import scipy as sp
from pysteps.noise.fftgenerators import initialize_nonparam_2d_ssft_filter
from pysteps.noise.fftgenerators import generate_noise_2d_ssft_filter
from scipy.stats import pearsonr
import random


#==============================================================================

# function to calculate lag-1 temporal correlation between most recent precip fields
def getAlpha(data0,data1):
    
    # get average temporal correlation of field at timestep t
    lp, _ = pearsonr(data1.flatten(),data0.flatten())
    
    # optional - get correlation of nonzero portions of precipitation fields only
    # lp, _ = pearsonr(data1[data1+data0>0.].flatten(),data0[data1+data0>0.].flatten())
    
    return(lp)


# =============================================================================
# distance between two lat/lon coordinates, in meters

def latlondistance(lat1,lon1,lat2,lon2):     # it probably more precise [it is definitly not a plane]
    R=6371.
    dlat=np.radians(lat2-lat1)
    dlon=np.radians(lon2-lon1)
    a=np.sin(dlat/2.)*np.sin(dlat/2.)+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.)*np.sin(dlon/2.);
    c=2.*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return R*c*1000.


# =============================================================================


def getCorrNoiseAR1(n, h, dt,obsFile,windFile,seednum):
    
    # ---- grab hourly MERRA2 wind data at 850 mb over 35N-45N, 85W-100W [m/s]
    # MERRA2 files were downloaded from GES-DISC and regridded
    dsw = Dataset(windFile)
    
    # get number of hours between start of file and date dt at hour h
    d_start = num2date(dsw.variables['time'][0],dsw.variables['time'].units)
    ndw = 24*(dt - date(d_start.year,d_start.month,d_start.day)).days + h
    #print('Wind file start: ',d_start,ndw)
    
    
    # ----  open hourly IMERG data  ----
    ds = Dataset(obsFile)
    dslon=ds['longitude'][:]
    dslat=ds['latitude'][:]
    
    dlon=np.median(dslon[1:]-dslon[0:-1])
    dlat=np.median(dslon[1:]-dslon[0:-1])

    
    # get number of hours between start of file and date dt at hour h
    d_start = num2date(ds.variables['time'][0],ds.variables['time'].units)
    nd = 24*(dt - date(d_start.year,d_start.month,d_start.day)).days + h
    print(nd)
    print(h)
    print(dt)
    print(d_start)
    #print('IMERG file start: ',d_start,nd)
    
    imerg = ds.variables['prcp'][:,:,nd].astype('float32')
    imerg[imerg<0.1] = 0.
    
    xsize=imerg.shape[1]
    ysize=imerg.shape[0]
    
    
    yind=np.repeat(np.arange(0,ysize),xsize).reshape(ysize,xsize).astype('int16')
    xind=np.tile(np.arange(0,xsize),ysize).reshape(ysize,xsize).astype('int16')
    
    
    tdy=latlondistance(dslat[0], dslon[0], dslat[0]+dlat, dslon[0])
    dy=np.empty_like(imerg)
    dy[:]=tdy
    gridx,gridy=np.meshgrid(dslon,dslat) 
    dx=latlondistance(gridy.flatten(), gridx.flatten(), gridy.flatten(), gridx.flatten()+dlon)
    dx=dx.reshape(ysize,xsize)
    
    
    # create array to hold simulated noise
    s = np.empty((n,ysize,xsize),dtype=np.float32)
    
    
    # --- SIMULATE FIRST FIELD ---
    # you could change winsize to be 64/128
    winsize=128
    
    if len(imerg[imerg>=0.1]) > 0.05*ysize*xsize:          
        # only use pysteps smoothing filter if there is actually rain in the study area
        print('Replicating power spectrum of first IMERG field')
        
        Fnp = initialize_nonparam_2d_ssft_filter(imerg,win_size=(winsize,winsize)) 
        s[0,:,:] = generate_noise_2d_ssft_filter(Fnp,seed=seednum)
        
        # s[0,:,:] = splitPysteps(imerg, winsize, seednum)
        firstRain=True
        recentRain = np.copy(imerg)

    else:
        # otherwise just use smoothed white noise
        s[0,:,:] = np.random.normal(size=(ysize,xsize))#,seed=seednum)
        firstRain=False
    
    
    last_rn = s[0,:,:] # record this instance of random noise as most recent instance of random noise
    
    
    # set initial value of alpha parameter - determines degree to which noise is incorporated into field at each timestep
    alpha = 0.95
    
    ts = 60
    
    # --- ADVECT FIELD FORWARD IN TIME ---
    print('Running Semi-Lagrangian Scheme')
    print(n)
    for hr in range(1,n):
        
        
        # get wind values from PREVIOUS hour and calculate dx, dy that correspond to this time step
        u = dsw.variables['U850'][:,:,ndw+int(hr*60/ts)-1] # eastward wind [m/s]
        
        deltax = u*60.*ts # distance wind travels each time step in positive x dir [m]
        dix = deltax//dx # number of grid steps to shift in semi-lagrangian scheme in units of [0.1 deg pixels]
        
        
        v = dsw.variables['V850'][:,:,ndw+int(hr*60/ts)-1] # northward wind [m/s]
        deltay = -v*60.*ts # distance wind travels each time step in positive y dir [m]
        diy = deltay//dy # number of grid steps to shift in semi-lagrangian scheme [0.1 deg pixels]
        print(hr)
        print(nd+int(hr*60/ts))
        imergh = ds.variables['prcp'][:,:,nd+int(hr*60/ts)].astype('float32')
        
        
        # use pySTEPS to generate noise with power spectrum of IMERG field
        # only use pysteps smoothing filter if there is actually rain in the study area
        
        if len(imergh[imergh>=0.1]) > 0.05*ysize*xsize: # atleast 5% of study area must be rainy
            
            Fnp = initialize_nonparam_2d_ssft_filter(imergh,win_size=(winsize,winsize))
            rn = generate_noise_2d_ssft_filter(Fnp,seed=seednum+hr)
    
            # rn = splitPysteps(imergh, winsize, seednum+hr) # to apply pysteps over windows
            
            # get average temporal correlation of field in previous timestep
            alpha = getAlpha(imergh,recentRain)
           
            firstRain=True
            recentRain = np.copy(imergh)
        
        
        else:
            
            # if field does not have at least 5% rainy pixels, generate random field 
            # using spatial correlation structure of last instance of rainfall 
            
            if firstRain==True:
                
                rn = generate_noise_2d_ssft_filter(Fnp,seed=seednum+hr)
                
                # rn = splitPysteps(recentRain, winsize, seednum+hr)
            
            else:
                # if first instance of a "rainy field" hasn't occurred yet in study period, use white noise
                rn = np.random.normal(size=(ysize,xsize))#,seed=seednum+hr) # white noise
        
        
        if np.isnan(rn[0,0])==True: # if pysteps smoothing process doesn't work for some reason, use last realization of smoothed noise
            
            rn = last_rn
            
        else:
            
            last_rn = rn
        
    
        # -- advect noise values from previous time step & perturb with noise field rn --
        ybefore=np.array((yind-diy)%ysize,dtype='int16')  #edge circulation detroyed...
        xbefore=np.array((xind-dix)%xsize,dtype='int16')
        s[hr,:] = (alpha*s[hr-1,ybefore,xbefore] + np.sqrt(1.-alpha**2)*rn).astype('float32')
        

   
     # return noise field
    return(s)


def generateNoise(n_ens,ts,dt,obsFile,windFile,newFile):
    
    hr = 0 # hour to start simulation at - we assume this is 0, but this can be changed    
    
    end_dt = dt + timedelta(hours=(hr+ts-1)) # end date of simulation
    
    
    print("Generating %d-member noise ensemble for %s to %s"%(n_ens,dt.strftime("%Y-%m-%d"),end_dt.strftime("%Y-%m-%d")))
    
    ds = Dataset(obsFile)
    dslon=ds['longitude'][:]
    dx=np.median(dslon[1:]-dslon[0:-1])
    xsize=dslon.shape[0]
    dslat=ds['latitude'][:]
    dy=np.median(dslon[1:]-dslon[0:-1])
    ysize=dslat.shape[0]
    
    # Create netcdf to write noise arrays into
    
    new_cdf = Dataset(newFile, 'w', format = "NETCDF4", clobber=True)
    
    # create array of time stamps
    time_hrs = [datetime(dt.year,dt.month,dt.day,0,0)+n*timedelta(hours=1) for n in range(ts)]
    units = 'hours since 1970-01-01 00:00:00 UTC'
    
    # create dimensions
    new_cdf.createDimension('lat', ysize)
    new_cdf.createDimension('lon', xsize)
    new_cdf.createDimension('ens_n', n_ens)
    new_cdf.createDimension('time', ts)
    
    # write time stamps to variable
    time = new_cdf.createVariable('time','d', ('time'))
    time.units = units
    time[:] = date2num(time_hrs,units,calendar="gregorian")
    
    # add lat, and lon variables
    latitude = new_cdf.createVariable('latitude', 'f4', ('lat'), zlib=True,least_significant_digit=2)
    latitude.units = 'degrees_north'
    latitude.long_name = 'latitude'
    # open input precipitation netcdf to get latitude and longitude arrays
    
    latitude[:] = ds['latitude'][:]
    
    longitude = new_cdf.createVariable('longitude', 'f4', ('lon'), zlib=True,least_significant_digit=2)
    longitude.units = 'degrees_east'
    longitude.long_name = 'longitude'
    longitude[:] = ds['longitude'][:]

    ds = None
    
    noise = new_cdf.createVariable('q', 'f4', ('ens_n','time','lat','lon'), zlib=True,least_significant_digit=4)
    noise.units = '--'
    noise.long_name = 'Uniform Noise'
    
    
    for nn in range(0,n_ens):
        
        # retrieve instance of correlated white noise for ts timesteps starting at date dt and hour hr
        # each ensemble from different seed
        s = getCorrNoiseAR1(ts,hr,dt,obsFile,windFile,seednum=random.randint(1,10000))
        
        s = 0.5*(1+sp.special.erf((s/math.sqrt(2)))) # convert to uniform noise
        
        noise[nn,:,:,:] = s # store in netcdf noise array
        
        print("Noise ensemble member",nn,"complete.")
        
    
    new_cdf.close()




