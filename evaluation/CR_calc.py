# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:16:49 2022

Containing Ratio: CR is a popular method to evaluate ensemble performance

Input: 
    IMERG precipitation to be compared with STREAM-Sat precipitation ensembles
    STREAM-Sat precipitation ensemble
    "ground truth" StageIV is used here

Output；
    CR in four categories hits, missies, false alarms, correct negetive.
    Classification is based on IMERG-StageIV to evaluate STREAM-Sat performance 
    condition on different IMERG errors.
    CR can only be calculated for STREAM-Sat ensembles
    
@author: Kaidi Peng
"""

from netCDF4 import Dataset,num2date
from datetime import date
import numpy as np


# the number of ensembles
ens=20

dtstart = date(2017,8,1)
dtend = date(2017,8,31)

obsFname = "IMERGE.hourly.nc"
obsds = Dataset(obsFname)
dur=(dtend-dtstart).days +24

imergobs= obsds.variables['prcp'][:,:,:] 

# 0.1 mm/hr is raining threshold
imergobs[imergobs<0.1] = 0.
imgPrcp=np.round(imergobs,2)
#=====================================================================

STREAMDPRFname = "STREAM_20170801_20170831.nc" 
ensDPRds = Dataset(STREAMDPRFname)
simDPRPrcp= ensDPRds.variables['prcp'][:,:,:,:]  #lat-lon-time-ens
#=============================================================================

STAFname = "StageIV_0.1deg.hourly.nc"
stads = Dataset(STAFname)
staobs=stads.variables['prcp'][:,:,:] 

staobs[staobs<0.1] = 0.
staPrcp=np.round(staobs,2)  

#=======================================================================

ts=np.size(imgPrcp,2)
xsize=np.size(imgPrcp,1)
ysize=np.size(imgPrcp,0)



maxsimP=np.max(simDPRPrcp,3) 
minsimP=np.min(simDPRPrcp,3)   # lat lon ts

CN_all=((staPrcp>=minsimP) & (staPrcp<=maxsimP))
lenall=np.sum((imgPrcp>=0),2)
CR_all=np.nanmean(np.sum(CN_all,2)/lenall)

# CN_all[(staPrcp==0) | (imgPrcp==0)]=np.nan
# CR_hit=np.nanmean(CN_all,2)+0.0

   
#======== hits ==========     
Referhit=staPrcp[(staPrcp>0) & (imgPrcp>0)]
minsimP_hit=minsimP[(staPrcp>0) & (imgPrcp>0)]
maxsimP_hit=maxsimP[(staPrcp>0) & (imgPrcp>0)]

if len(Referhit)>0:
    CR_hit=np.sum((Referhit>=minsimP_hit) & (Referhit<=maxsimP_hit))/len(Referhit)

    
#======== misses ==========     
Refermiss=staPrcp[(imgPrcp==0) & (staPrcp>0)]
minsimP_miss=minsimP[(staPrcp>0) & (imgPrcp==0)]
maxsimP_miss=maxsimP[(staPrcp>0) & (imgPrcp==0)]

if len(Refermiss)>0:
    CR_miss=np.sum((Refermiss>=minsimP_miss) & (Refermiss<=maxsimP_miss))/len(Refermiss)

    
#======== false alarms ==========     
Referfalarm=staPrcp[(imgPrcp>0) & (staPrcp==0)]
minsimP_falarm=minsimP[(staPrcp==0) & (imgPrcp>0)]
maxsimP_falarm=maxsimP[(staPrcp==0) & (imgPrcp>0)]


if len(Referfalarm)>0:
    CR_falarm=np.sum((Referfalarm>=minsimP_falarm) & (Referfalarm<=maxsimP_falarm))/len(Referfalarm)
    
    
#======== correct negetive ==========     
Referconodet=staPrcp[(imgPrcp==0) & (staPrcp==0)]
minsimP_conodet=minsimP[(staPrcp==0) & (imgPrcp==0)]
maxsimP_conodet=maxsimP[(staPrcp==0) & (imgPrcp==0)]


if len(Referconodet)>0:
    CR_conodet=np.sum((Referconodet>=minsimP_conodet) & (Referconodet<=maxsimP_conodet))/len(Referconodet)

lenall=len(Referhit)+len(Refermiss)+len(Referfalarm)+len(Referconodet)

#==========output=================================================
print('%s'%(STREAMDPRFname ))
print("mean hits CR: %0.4f --- %0.4f"%(CR_hit,len(Referhit)/lenall))
print("mean misses CR: %0.4f --- %0.4f"%(CR_miss,len(Refermiss)/lenall))
print("mean false alarm CR: %0.4f --- %0.4f"%(CR_falarm,len(Referfalarm)/lenall))
print("mean correct non-detect CR: %0.4f --- %0.4f"%(CR_conodet,len(Referconodet)/lenall))
print("mean CR: %0.4f and %0.4f"%(CR_all,lenall))

# save the spatial map of average CR
np.save('CR_all',np.sum(CN_all,2)/lenall)
#----------------------------------------------------------------------------
