# StatMeanderpy

<p align="center">
  <img src="https://github.com/snohatech/StatMeanderpy/blob/main/ReadMeFigs/1bend.gif" alt="GIF 1" width="22.5%" />
  <img src="https://github.com/snohatech/StatMeanderpy/blob/main/ReadMeFigs/2bend.gif" alt="GIF 1" width="22.5%" />
  <img src="https://github.com/snohatech/StatMeanderpy/blob/main/ReadMeFigs/3bend.gif" alt="GIF 1" width="22.5%" />
  <img src="https://github.com/snohatech/StatMeanderpy/blob/main/ReadMeFigs/4bend.gif" alt="GIF 1" width="22.5%" />
</p>

StatMeanderpy is a deriviation of Meanderpy (https://github.com/zsylvester/meanderpy) developed by Zoltán Sylvester, which  implements a simple numerical model of river meandering, the one described by Howard & Knutson in their 1984 paper ["Sufficient Conditions for River Meandering: A Simulation Approach"](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/WR020i011p01659).

River migration can be predicted using a similar approach to weather forecasting. By employing probabilistic modeling techniques, we have developed MCMC-based river migration risk maps. For more details, you can refer to our [EGU paper here](https://meetingorganizer.copernicus.org/EGU23/EGU23-17240.html) conference paper here.

## Usage

First, import the library:

```
import numpy as np
import os
import matplotlib.pyplot as plt
import meanderpy as mp
```

We first need to define the river channel centerline. For this example, we are using the Ucayali River centerline collected for 40 years using [RivMAP](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016EA000196). Centerline data are available in [ChannelGeometry](https://github.com/snohatech/StatMeanderpy/tree/main/ChannelGeometry) folder. 

```ruby
os.chdir(r"/Users/StatMeanderpyGithub/ChannelGeometry/1Bend")
cl1=np.loadtxt('0year.txt',delimiter=' ')
cl2=np.loadtxt('10year.txt',delimiter=' ')

xin = cl1[:,1][::-1]
yin = cl1[:,0][::-1]
xfi= cl2[:,1][::-1]
yfi = cl2[:,0][::-1]
plt.plot(xin,yin,label = 'Initial Channel')
plt.plot(xfi,yfi,label = 'Final Channel')
```

Defining parameters:

```ruby
nit = 100                    # number of iterations
W = 100.0                    # channel width (m)
D = 10.0                      # channel depth (m)
depths = D * np.ones((nit,))  # channel depths for different iterations  
pad = 0# padding (number of nodepoints along centerline)
deltas = 50.0                # sampling distance along centerline           
Cfs = 0.02 * np.ones((nit,))
crdist = 1.8 * W               # threshold distance at which cutoffs occur
kl = 200/(365*24*60*60.0)   # migration rate constant (m/s)
kv =  1.0e-12               # vertical slope-dependent erosion rate constant (m/s)
dt = 0.1*(365*24*60*60.0)      # time step (s)
dens = 1000                  # density of water (kg/m3)
saved_ts = 1                # which time steps will be saved
Sl = 0.0                     # initial slope (matters more for submarine channels than rivers)
t1 = 0                    # time step when incision starts
t2 = 0                    # time step when lateral migration starts
t3 = 0     
aggr_factor = 2e-9            # aggradation factor (m/s, about 0.18 m/year, it kicks in after t3) after t3)
sc = 1.0
y=cl1[:,0]*10
x=cl1[:,1]*10
z=np.zeros(len(x))
H=depths[0]
```
