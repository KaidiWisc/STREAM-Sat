# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:16:49 2022

The continuous ranked probability skill score : CRPS
is another popular method to evaluate ensemble performance

Reference:
    Thorarinsdottir, T. L., Gneiting, T., & Gissibl, N. (2013). 
    Using Proper Divergence Functions to Evaluate Climate Models. 
    SIAM/ASA Journal on Uncertainty Quantification, 
    1(1), 522-534. doi:10.1137/130907550


Input: 
    IMERG precipitation to be compared with STREAM-Sat precipitation ensembles
    STREAM-Sat precipitation ensemble
    "ground truth" StageIV is used here

Output；
    CRPS in four categories hits, missies, false alarms, correct negetive.
    Classification is based on IMERG-StageIV to evaluate STREAM performance 
    condition on different IMERG errors.
    CRPS for STREAM-Sat ensembles are equivalent to MAE for IMERG
    
@author: Kaidi Peng
"""

from netCDF4 import Dataset
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
simDPRPrcp= ensDPRds.variables['prcp'][:,:,:,:]
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
tsize=ts
ensize=ens

#=========calc CRPS for STREAM-Sat MAE for IMERG=================================================
#---total--------------------------------------------------------
# delete all possible nan and infinity.
mask= (simDPRPrcp[:,:,:,1]>1e4) | (np.isnan(simDPRPrcp[:,:,:,1])==True) | (np.isfinite(simDPRPrcp[:,:,:,1])==False) | (staPrcp>1e4) | (np.isnan(staPrcp)==True) | (np.isfinite(staPrcp)==False)

staPrcp[mask==1]=np.nan 

part1=np.zeros((ysize,xsize,tsize))

for ensu in range(ensize):
    part1+=np.abs(simDPRPrcp[:,:,:,ensu]-staPrcp)  #ens

part11=part1/ensize  # ave on ensemble

part2=np.zeros((ysize,xsize,tsize))
for i in range(ensize):
    for j in range(ensize):
        part2=part2+np.abs(simDPRPrcp[:,:,:,i]-simDPRPrcp[:,:,:,j])

part=part11-part2/(2*ensize**2)  # yszie,xsize,tsize

CRPS_map=np.nanmean(part,2)
MAE_map=np.nanmean(np.abs(staPrcp-imgPrcp),2)


mask= (simDPRPrcp[:,:,:,1]<1e4) & (np.isnan(simDPRPrcp[:,:,:,1])==False) & (np.isfinite(simDPRPrcp[:,:,:,1])==True) & (staPrcp<1e4) & (np.isnan(staPrcp)==False) & (np.isfinite(staPrcp)==True)

#======== hits ==========     
Referhit=staPrcp[(staPrcp>0) & (imgPrcp>0) & mask]
imgPrcp_hit=imgPrcp[(staPrcp>0) & (imgPrcp>0) & mask]

enslen=len(Referhit)
ensPhit=np.zeros((enslen,ensize))
for i in range(ensize):
    ensi=simDPRPrcp[:,:,:,i]
    ensPhit[:,i]=ensi[(staPrcp>0) & (imgPrcp>0) & mask]
    
CRPSens_hit1=np.nanmean(np.abs(ensPhit-np.repeat(Referhit,ensize).reshape(enslen,ensize)),1)

CRPSens_hit2=np.zeros((enslen))
for i in range(ensize):
    for j in range(ensize):
            CRPSens_hit2=CRPSens_hit2+np.abs(ensPhit[:,i]-ensPhit[:,j])

CRPSens_hit=np.nanmean(CRPSens_hit1-CRPSens_hit2/(2*ensize**2))
MAE_hit=np.sum(np.abs(Referhit-imgPrcp_hit))/len(Referhit)
    
#======== misses ==========     
Refermis=staPrcp[(staPrcp>0) & (imgPrcp==0) & mask]
imgPrcp_mis=imgPrcp[(staPrcp>0) & (imgPrcp==0) & mask]

enslen=len(Refermis)
ensPmis=np.zeros((enslen,ensize))
for i in range(ensize):
    ensi=simDPRPrcp[:,:,:,i]
    ensPmis[:,i]=ensi[(staPrcp>0) & (imgPrcp==0) & mask]
    
CRPSens_mis1=np.nanmean(np.abs(ensPmis-np.repeat(Refermis,ensize).reshape(enslen,ensize)),1)

CRPSens_mis2=np.zeros((enslen))
for i in range(ensize):
    for j in range(ensize):
            CRPSens_mis2=CRPSens_mis2+np.abs(ensPmis[:,i]-ensPmis[:,j])

CRPSens_mis=np.nanmean(CRPSens_mis1-CRPSens_mis2/(2*ensize**2))
MAE_mis=np.sum(np.abs(Refermis-imgPrcp_mis))/len(Refermis)

#======== false alarms ==========     
Referfam=staPrcp[(staPrcp==0) & (imgPrcp>0) & mask]
imgPrcp_fam=imgPrcp[(staPrcp==0) & (imgPrcp>0) & mask]

enslen=len(Referfam)
ensPfam=np.zeros((enslen,ensize))
for i in range(ensize):
    ensi=simDPRPrcp[:,:,:,i]
    ensPfam[:,i]=ensi[(staPrcp==0) & (imgPrcp>0) & mask]
    
CRPSens_fam1=np.nanmean(np.abs(ensPfam-np.repeat(Referfam,ensize).reshape(enslen,ensize)),1)

CRPSens_fam2=np.zeros((enslen))
for i in range(ensize):
    for j in range(ensize):
            CRPSens_fam2=CRPSens_fam2+np.abs(ensPfam[:,i]-ensPfam[:,j])

CRPSens_fam=np.nanmean(CRPSens_fam1-CRPSens_fam2/(2*ensize**2))
MAE_fam=np.sum(np.abs(Referfam-imgPrcp_fam))/len(Referfam)
    
#======== correct negetive ==========     
Refercnd=staPrcp[(staPrcp==0) & (imgPrcp==0) & mask]
imgPrcp_cnd=imgPrcp[(staPrcp==0) & (imgPrcp==0) & mask]

enslen=len(Refercnd)
ensPcnd=np.zeros((enslen,ensize))
for i in range(ensize):
    ensi=simDPRPrcp[:,:,:,i]
    ensPcnd[:,i]=ensi[(staPrcp==0) & (imgPrcp==0) & mask]
    
CRPSens_cnd1=np.nanmean(np.abs(ensPcnd-np.repeat(Refercnd,ensize).reshape(enslen,ensize)),1)

CRPSens_cnd2=np.zeros((enslen))
for i in range(ensize):
    for j in range(ensize):
            CRPSens_cnd2=CRPSens_cnd2+np.abs(ensPcnd[:,i]-ensPcnd[:,j])

CRPSens_cnd=np.nanmean(CRPSens_cnd1-CRPSens_cnd2/(2*ensize**2))
MAE_cnd=np.sum(np.abs(Refercnd-imgPrcp_cnd))/len(Refercnd)
#--------------------------------------------------------------------------


lenall=len(Referhit)+len(Refermis)+len(Referfam)+len(Refercnd)

print('%s'%(STREAMDPRFname ))
print("mean hits CRPS: %0.4f --- %0.4f"%(CRPSens_hit,len(Referhit)/lenall))
print("mean misses CRPS: %0.4f --- %0.4f"%(CRPSens_mis,len(Refermis)/lenall))
print("mean false alarm CRPS: %0.4f --- %0.4f"%(CRPSens_fam,len(Referfam)/lenall))
print("mean correct non-detect CRPS: %0.4f --- %0.4f"%(CRPSens_cnd,len(Refercnd)/lenall))
print("mean CRPS: %0.4f and %0.4f"%(np.nanmean(CRPS_map),lenall))


print("mean hits MAE: %0.4f --- %0.4f"%(MAE_hit,len(Referhit)/lenall))
print("mean misses MAE: %0.4f --- %0.4f"%(MAE_mis,len(Refermis)/lenall))
print("mean false alarm MAE: %0.4f --- %0.4f"%(MAE_fam,len(Referfam)/lenall))
print("mean correct non-detect MAE: %0.4f --- %0.4f"%(MAE_cnd,len(Refercnd)/lenall))
print("mean MAE: %0.4f and %0.4f"%(np.nanmean(MAE_map),lenall))

# save CRPS/MAE map
np.save('CRPS_map',CRPS_map)
np.save('MAE_map',MAE_map)


