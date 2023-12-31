# STREAM-Sat
# STREAM-Sat: A Novel Near-Realtime Quasi-global Satellite-Only Ensemble Precipitation Dataset
Kaidi Peng et al.

This repository is to create near-realtime global precipitation ensembles that condition on satellite observations (e.g., IMERG: Integrated Multi-satellitE Retrievals for GPM; https://gpm.nasa.gov/data/imerg). We unified the methods proposed in Li et al. (2023) [doi:10.1109/tgrs.2023.3235270](https://ieeexplore.ieee.org/document/10011447) and Hartke et al. (2022) [https://doi.org/10.1029/2021WR031650](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021WR031650). We tried to solve the challenge of Near-Realtime (NRT) global precipitation generation due to the lack of ground-based gauge network and the complex error of satellite precipitation. 

The highlights of this method are 
1) no ground-based measurement is needed. The performance (e.g., ensemble spread and accuracy) is independent of gauge density;
2) It could be generated in Near-Realtime, which means its time latency is only affected by satellite products (e.g., IMERG Early has 4-hour latency);
3) It can be done globally, while the performance will be affected by satellite retrieval accuracy over different regions.

The inputs of STREAM-Sat are 
1) a gridded input precipitation dataset, such as IMERG Early; 
2) an error model for that input dataset at individual grid cell scale (CSGD in this example); 
3) motion vector: MERRA2 U850/V850 or IMERG's motion vector; 
4) covariates (optional): It depends on the covariates you used in the error model (WAR in this example);
The output is a user-defined number of precipitation and noise ensemble (20 in this example).

The performance of the ensemble can be evaluated by CR_calc.py and CRPS_calc.py.

Details about CSGD are on https://github.com/KaidiWisc/CSGD_error_model.git

The currently available STREAM-Sat data is over the year 2017.  
Peng, Kaidi (2023). Data from: STREAM-Sat: a novel near-realtime quasi-global satellite-only ensemble precipitation [Dataset]. Dryad. https://doi.org/10.5061/dryad.c59zw3rfk

We have a 12-page PowerPoint to summarize STREAM-Sat.
For any questions, feel free to contact Kaidi Peng (kaidi.peng@wisc.edu).
