# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:17:25 2020

This script uses CSGD error models trained over study area
to transform spatiotemporally correlation uniform noise into rainfall values
"""

from netCDF4 import Dataset,num2date
from datetime import date, timedelta, datetime
import numpy as np
import scipy as sp

#==============================================================================
# Function to retrieve wetted area ratio (WAR) across entire field
# - more efficient than retrieving pixel by pixel


area_set = False
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
    
def getWARfield(field,r):
    
    ysize = np.shape(field)[0]
    xsize = np.shape(field)[1]
    
    if area_set == False:
        set_area(ysize,xsize)
    
    rainy = np.zeros(np.shape(field))
    # default rain threshold: 0.1mm/hr
    # when the observation > 0.1mm/hr, it is considered as rain
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


# ----------------------------------------------------------------------------

def simulatePrecip(dt,n_ens,ts,obsFile,noiseFile,paramsFile,CSGDW,verbose=False):
    
    end_dt = dt + timedelta(hours=(ts-1)) # end date of simulation
    print("Generating %d-member precip ensemble for %s - %s"%(n_ens,dt.strftime("%Y-%m-%d"),end_dt.strftime("%Y-%m-%d")))
    
    # ---  indicate whether to use correlated noise or not  ---
    corr = "imerg"   # corr can equal "imerg" or "none"
    
    
    # ---------------------  READ IN SATELLITE PRECIPITATION  -----------------
    i1 = (dt - date(dt.year,1,1)).days*24  # starting index of IMERG data for simulation period
    i2 = i1 + ts   # ending index
    
    ds = Dataset(obsFile)
    print(i1)
    print(i2)
    obs = ds.variables['prcp'][:,:,i1:i2].astype('float32') # grab IMERG data from simulation period
    obs[obs<0.1] = 0.
    
    ysize = np.shape(obs)[0]
    xsize = np.shape(obs)[1]
    
    
    # ------ open up wetted area ratio covariate data for simulation period ----
    # ds = Dataset(wd + 'data/WetAR%d.r10.hourly.nc'%dt.year)
    # war = ds.variables['war'][:,:,i1:i2].astype('float32')
    
    war = np.zeros(np.shape(obs))
    
    for i in range(ts):
        
        war[:,:,i] = getWARfield(obs[:,:,i],10)
    
    
    # -------------------------  RETRIEVE NOISE  ------------------------------
    
    if corr=="imerg":
        
        # ----  open correlated noise file  ----
        ds = Dataset(noiseFile)
        # find index of starting date for precip simulation
        ds_dts = num2date(ds.variables['time'][:],units=ds.variables['time'].units,
                              only_use_cftime_datetimes=False)
        
        ds_dts = np.array(list(datetime.strptime(str(d),'%Y-%m-%d %H:%M:%S') for d in ds_dts))
        
        i_start = int(np.where(ds_dts==datetime(dt.year,dt.month,dt.day))[0][:])
        i_end = i_start + ts
        
        q = ds.variables['q'][:n_ens,i_start:i_end,:,:].astype('float16')
        print(i_start,i_end,np.shape(q))
        # Note: if no correlated noise exists for this time range, create some and 
        # save to netcdf by calling generateNoiseEnsemble.py
    
    
    elif corr=="none":
        
        # -----  generate uncorrelated (white) noise  -----
        q = np.random.uniform(size=(n_ens,ts,ysize,xsize)).astype('float16')
    
    
    
    # set upper threshold for uniform noise so that unrealistically extreme
    # values don't get selected from conditional distribution tails
    q[q>0.999] = 0.999
    
    
    
    
    # ---------- Read in parameter grids for precip error model ---------------       
    
    ds = Dataset(paramsFile)
    
    lin = False # indicate if error model uses linear or nonlinear regression
    
    climparams = np.empty((3,ysize,xsize))
    n=0
    for name in ('clim1','clim2','clim3'):
        climparams[n,:,:] = ds.variables[name][:,:]
        n+=1
    
    reg = np.empty((6,ysize,xsize))
    n=0
    for name in ('par1','par2','par3','par4','par5'):
        reg[n,:,:] = ds.variables[name][:,:]
        n+=1
    
    reg[5,:,:] = 0.
    
    imean = ds.variables['mean'][:,:]
    warmean = ds.variables['WARmean'][:,:]
    
    ds = None
    
    
    
    # -------------------- SIMULATE PRECIPITATION --------------------------------
    
    
    simPrcp = np.zeros(np.shape(q),dtype=np.float32) # create empty array to hold simulated precip
    
    
    # -- loop through each pixel in study area --
    for y in np.arange(0,ysize-CSGDW+1,CSGDW):
        
        for x in np.arange(0,xsize-CSGDW+1,CSGDW):
            
            y1 = y # starting y index of subset window
            y2 = y+CSGDW # ending y index of subset window
            x1 = x # starting x index of subset window
            x2 = x+CSGDW # ending x index of subset window
        
            obs_0 = obs[y1:y2,x1:x2,:]  # (w,w,ts)
            obs_1=np.zeros((1,CSGDW,CSGDW,ts))  #(1,w,w,ts)
            obs_1[0,:,:,:]=obs_0    #(1,w,w,ts)
            obs_=obs_1.transpose(0,3,1,2)   #(1,ts,w,w)
            war_0 = war[y1:y2,x1:x2,:]   
            war_1=np.zeros((1,CSGDW,CSGDW,ts))  #(1,w,w,ts)
            war_1[0,:,:,:]=war_0    #(1,w,w,ts)
            war_=war_1.transpose(0,3,1,2)   #(1,ts,w,w)
            qVals = q[:,:,y1:y2,x1:x2]  # n_ens,ts,ysize,xsize
            
            # 10 by 10 grid cells share the same parameters. Select one from 10 by 10 is OK.
            mu_clim=climparams[:,y1+1,x1+1][0]   
            sigma_clim=climparams[:,y1+1,x1+1][1]
            delta_clim=climparams[:,y1+1,x1+1][2]

            # note: this is currently using the equivalent of pcsgd(0.0,condparams0) when IMERG and WAR is zero, rather than pcsgd(0.1,condparams0)
            logarg=reg[:,y1+1,x1+1][1] + reg[:,y1+1,x1+1][2]*obs_/imean[y1+1,x1+1] + reg[:,y1+1,x1+1][4]*war_/warmean[y1+1,x1+1]
            mu=mu_clim/reg[:,y1+1,x1+1][0]*np.log1p(np.expm1(reg[:,y1+1,x1+1][0])*logarg)
            sigma=reg[:,y1+1,x1+1][3]*sigma_clim*np.sqrt(mu/mu_clim)        
            delta=delta_clim
            
            quants=delta+sp.stats.gamma.ppf(qVals,np.power(mu/sigma,2),scale=np.power(sigma,2)/mu,loc=0)
            quants[quants<0.]=0.
            simPrcp[:,:,y1:y2,x1:x2]=quants

                                
    
    # -------- Apply precipitation detection threshold ------------------------
    # all precip less than 0.1 mm/hr is considered zero precip
    simPrcp[simPrcp<0.1] = 0.

    return(simPrcp)




