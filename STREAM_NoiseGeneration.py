  # -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:13:43 2020

@author: Sam Hartke

Script to generate and save netcdf of ensemble of correlated uniform noise using
pysteps replication of local spatial structures and semi-lagrangian scheme
based on MERRA2 wind fields

"""

from netCDF4 import Dataset, date2num, num2date
import numpy as np
from datetime import date, timedelta, datetime
import math
import scipy as sp
from tqdm import tqdm
from pysteps.noise.fftgenerators import initialize_nonparam_2d_ssft_filter
from pysteps.noise.fftgenerators import generate_noise_2d_ssft_filter
from scipy.stats import pearsonr
import random
from pysteps import utils

# from pysteps.timeseries.autoregression import estimate_ar_params_yw


#==============================================================================

def _get_mask(Size, idxi, idxj, win_fun= "tukey"):
    """Compute a mask of zeros with a window at a given position."""

    idxi = np.array(idxi).astype(int)
    idxj = np.array(idxj).astype(int)

    win_size = (idxi[1] - idxi[0], idxj[1] - idxj[0])
    if win_fun is not None:
        wind = utils.tapering.compute_window_function(win_size[0], win_size[1], win_fun)
        wind += 1e-6  # avoid zero values

    else:  # i am not using square window, I am using default windown setting turkey!!!
        wind = np.ones(win_size)

    mask = np.zeros(Size)
    mask[idxi.item(0) : idxi.item(1), idxj.item(0) : idxj.item(1)] = wind
    
    return mask

def spatial_getAlpha(field1,field2,win_size,overlap=0.3):
    
    ysize=np.shape(field1)[0]
    xsize=np.shape(field1)[1]
    
    Alpha_matrix=np.zeros(np.shape(field1))
    
    dim=[ysize,xsize]
    dim_x = xsize
    dim_y = ysize

    # prepare indices
    idxi = np.zeros(2, dtype=int)
    idxj = np.zeros(2, dtype=int)

    # number of windows
    num_windows_y = np.ceil(float(dim_y) / win_size[0]).astype(int)
    num_windows_x = np.ceil(float(dim_x) / win_size[1]).astype(int)

    # and allocate it to the final grid
    F = np.zeros((num_windows_y, num_windows_x, ysize, xsize)) # a indicator
    sM = np.zeros((ysize, xsize))  # recording of assigmnet times
    sumF=np.zeros((ysize, xsize))
    aal0=0.5
    # loop rows
    for i in range(F.shape[0]):
        # loop columns
        for j in range(F.shape[1]):

            # compute indices of local window
            idxi[0] = int(np.max((i * win_size[0] - overlap * win_size[0], 0)))
            idxi[1] = int(
                np.min((idxi[0] + win_size[0] + overlap * win_size[0], dim_y))
            )
            idxj[0] = int(np.max((j * win_size[1] - overlap * win_size[1], 0)))
            idxj[1] = int(
                np.min((idxj[0] + win_size[1] + overlap * win_size[1], dim_x))
            )

            # build localization mask
            # TODO: the 0.01 rain threshold must be improved
            mask = _get_mask(dim, idxi, idxj, win_fun="tukey")
            mask1=mask.copy()
            maskn=mask.copy()
            mask1[mask1>0]=1
            maskn[maskn>0]=1
            maskn[maskn<=0]=np.nan
            mask1[mask1<=0]=0
            aal=getAlpha(field1*maskn,field2*maskn)
      
            if np.isnan(aal)==True:
                aal=aal0
            else:
                aal0=aal
            Alpha_matrix+=aal*mask1
            print(aal)
            sumF+=mask1
    
    Alpha_matrix=Alpha_matrix/sumF
    
    return Alpha_matrix

#==============================================================================

# function to calculate lag-1 temporal correlation between most recent precip fields
def getAlpha(data0,data1):
    
    dat0=data0[(np.isnan(data0)==False) & (np.isnan(data1)==False)]
    dat1=data1[(np.isnan(data0)==False) & (np.isnan(data1)==False)]
    
    # get average temporal correlation of field at timestep t
    lp, _ = pearsonr(dat1.flatten(),dat0.flatten())
    
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

# #of time steps, start time
def getCorrNoiseAR1(n, dt,obsFile,windFile,seednum,tres):
    
    # ---- grab hourly MERRA2 wind data at 850 mb over 35N-45N, 85W-100W [m/s]
    # MERRA2 files were downloaded from GES-DISC and aggregated to yearly files in writeWindNetcdfs.py
    dsw = Dataset(windFile)
    
    # get number of hours between start of file and date dt at hour h
    d_start = num2date(dsw.variables['time'][0],dsw.variables['time'].units)
    ndw = int(24/tres)*(dt - date(d_start.year,d_start.month,d_start.day)).days 
    #print('Wind file start: ',d_start,ndw)
    
    
    # ----  open hourly IMERG data  ----
    ds = Dataset(obsFile)
    dslon=ds['longitude'][:]
    dslat=ds['latitude'][:]
    
    dlon=np.median(dslon[1:]-dslon[0:-1])
    dlat=np.median(dslon[1:]-dslon[0:-1])

    
    # get number of hours between start of file and date dt at hour h
    d_start = num2date(ds.variables['time'][0],ds.variables['time'].units)
    nd = int(24/tres)*(dt - date(d_start.year,d_start.month,d_start.day)).days 
    print(nd)
    print(dt)
    print(d_start)  # 2017-09-01 
    #print('IMERG file start: ',d_start,nd)
    
    imerg = ds.variables['prcp'][:,:,nd].astype('float32')
    imerg[imerg<0.1] = 0.
    
    xsize=imerg.shape[1]
    ysize=imerg.shape[0]
    
    
    yind=np.repeat(np.arange(0,ysize),xsize).reshape(ysize,xsize).astype('int16')
    xind=np.tile(np.arange(0,xsize),ysize).reshape(ysize,xsize).astype('int16')
    
    
    # create array to hold simulated noise
    s = np.empty((n,ysize,xsize),dtype=np.float32)
    
    
    # --- SIMULATE FIRST FIELD ---
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
        recentRain = np.copy(imerg)
    
    last_rn = s[0,:,:] # record this instance of random noise as most recent instance of random noise
    
    
    # set initial value of alpha parameter - determines degree to which noise is incorporated into field at each timestep
    alpha = 0.95
    
    ts = 60
    
    # --- ADVECT FIELD FORWARD IN TIME ---
    print('Running Semi-Lagrangian Scheme')
    print(n)
    for hr in range(1,n):
        
        
        # get wind values from PREVIOUS hour and calculate dx, dy that correspond to this time step
        dix= dsw.variables['U_MV'][:,:,ndw+hr-1] # eastward wind [m/s]      
        
        diy = -dsw.variables['V_MV'][:,:,ndw+hr-1] # northward wind _> SOUTHWARD

        # print(ndw+hr-1)
        imergh = ds.variables['prcp'][:,:,nd+hr].astype('float32')
        
        
        # use pySTEPS to generate noise with power spectrum of IMERG field
        # only use pysteps smoothing filter if there is actually rain in the study area
        
        if len(imergh[imergh>=0.1]) > 0.05*ysize*xsize: # atleast 5% of study area must be rainy
            
            Fnp = initialize_nonparam_2d_ssft_filter(imergh,win_size=(winsize,winsize))
            rn = generate_noise_2d_ssft_filter(Fnp,seed=seednum+hr)
    
            # rn = splitPysteps(imergh, winsize, seednum+hr) # to apply pysteps over windows
            
            # get average temporal correlation of field in previous timestep
            alpha =spatial_getAlpha(imergh,recentRain,win_size=(winsize,winsize))
           
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
        ybefore=np.array((np.round(yind-diy))%ysize,dtype='int16')  #edge circulation detroyed...
        xbefore=np.array((np.round(xind-dix))%xsize,dtype='int16')
        s[hr,:] = (alpha*s[hr-1,ybefore,xbefore] + np.sqrt(1.-alpha**2)*rn).astype('float32')
        # time lat-lon

   
     # return noise field
    return(s)

                # nEns,tsi,starti
def generateNoise(n_ens,ts,dt,obsFile,windFile1,windFile2,newFile,tres):
    
    
    end_dt = dt + timedelta(hours=int(ts-1)*tres) # end date of simulation , ts is hh
    
    
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
    time_hrs = [datetime(dt.year,dt.month,dt.day,0,0)+n*timedelta(hours=tres) for n in range(ts)]
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
        s = getCorrNoiseAR1(ts,dt,obsFile,windFile1,seednum=random.randint(1,10000),tres=tres)
        
        s = 0.5*(1+sp.special.erf((s/math.sqrt(2)))) # convert to uniform noise
        
        noise[nn,:,:,:] = s # store in netcdf noise array
        
        print("Noise ensemble member",nn,"complete.")
        
    
    new_cdf.close()




