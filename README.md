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
import pandas as pd
from scipy import interpolate          
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
nit = 100                         # number of iterations
W = 100.0                         # channel width (m)
D = 10.0                          # channel depth (m)
depths = D * np.ones((nit,))      # channel depths for different iterations  
pad = 0                           # padding (number of nodepoints along centerline)
deltas = 50.0                     # sampling distance along centerline           
Cfs = 0.02 * np.ones((nit,))      #chezy friction factor
crdist = 1.8 * W                  # threshold distance at which cutoffs occur
kl = 200/(365*24*60*60.0)         # migration rate constant (m/s)
kv =  1.0e-12                     # vertical slope-dependent erosion rate constant (m/s)
dt = 0.1*(365*24*60*60.0)         # time step (s)
dens = 1000                       # density of water (kg/m^3)
saved_ts = 1                      # which time steps will be saved
Sl = 0.0                          # initial slope (matters more for submarine channels than rivers)
t1 = 0                            # time step when incision starts
t2 = 0                            # time step when lateral migration starts
t3 = 1200                         # time step when aggradation starts
aggr_factor = 2e-9                # aggradation factor (m/s, about 0.18 m/year, it kicks in after t3) after t3)
x=cl1[:,1]*10                     # initial x channel geometry
y=cl1[:,0]*10                     # initial y channel geometry
z=np.zeros(len(x))                # initial z (0)
H=depths[0]                       # no height in channel
```

Meanderpy will use the initial channel geometry to migrate the channel with the defined parameters. 

```ruby
ch=mp.Channel(-x,-y,z,W,H)
chb=mp.ChannelBelt(channels=[ch], cutoffs=[], cl_times=[0.0], cutoff_times=[])
chb.migrate(nit,saved_ts,deltas,pad,crdist,depths,Cfs,kl,kv,dt,dens,t1,t2,t3,aggr_factor) # channel migration
channel_coordinate = pd.DataFrame({"x":chb.channels[nit-1].x, "y":chb.channels[nit-1].y, "Z":chb.channels[nit-1].z}) # the way it work is this: chb.channels[i].x will give you the x coordinates for the ith time step

simx = 0.1*chb.channels[np.int(nit-1)].x
simy = 0.1*chb.channels[np.int(nit-1)].y

ax = pd.DataFrame({"x":simx, "y":simy}).plot.line(x='x', y='y', label='Simulation Result')
pd.DataFrame({"x":-cl1[:,1], "y":-cl1[:,0]}).plot.line(x='x', y='y', ax= ax, label='Initial Channel (0 years)')
pd.DataFrame({"x":-cl2[:,1], "y":-cl2[:,0]}).plot.line(x='x', y='y', ax= ax, label='Final Channel (10 years)')
```

Because the channel array data do not have the same amount of array lengths, we need to interpolate them to have the same length. We are interpolating for 1000 lengths for initial (ch1) and final (ch2) channel centerlines. 

```ruby
# Interpolating the initial channel
t = np.linspace(0,1,np.shape(cl1[:,])[0]) 
x_o = -cl1[:,1].flatten()           
y_o = -cl1[:,0].flatten()         
fx_o = interpolate.interp1d(t,x_o)     
fy_o = interpolate.interp1d(t, y_o)    
tnew = np.linspace(0,1,1000)   
xnew_o = fx_o(tnew) + np.random.normal(0,0.1,1000)  
ynew_o = fy_o(tnew) + np.random.normal(0,0.1,1000)
xnew_o = fx_o(tnew)   # get interpolated x values
ynew_o = fy_o(tnew)   # get interpolated y values
data_obs_ins = np.array([xnew_o,ynew_o])
data_obs_ins = np.round(data_obs_ins, 2) 
#data_obs_ins =np.flip(data_obs_ins, axis=1)
# Interpolating the final channel
t = np.linspace(0,1,np.shape(cl2[:,])[0]) 
x_o = -cl2[:,1].flatten()          
y_o = -cl2[:,0].flatten()           
fx_o = interpolate.interp1d(t,x_o)    
fy_o = interpolate.interp1d(t, y_o)    
tnew = np.linspace(0,1,1000)          
xnew_o = fx_o(tnew)                   
ynew_o = fy_o(tnew)
data_obs = np.array([xnew_o,ynew_o])
data_obs = np.flip(data_obs, axis=1)
data_obs = np.round(data_obs, 1)
# modelled
# Interpolating the simulated channel
t = np.linspace(0,1,np.shape(0.1*chb.channels[np.int(nit-1)].x)[0])
x_m = 0.1*chb.channels[np.int(nit-1)].x
y_m = 0.1*chb.channels[np.int(nit-1)].y
fx_m = interpolate.interp1d(t,x_m)
fy_m = interpolate.interp1d(t, y_m)
tnew = np.linspace(0,1,1000)
xnew_m = fx_m(tnew)
ynew_m = fy_m(tnew)
ynew_ms = ynew_m+np.random.normal(loc=0.0, scale= 0.1, size=1)
xnew_ms = xnew_m+np.random.normal(loc=0.0, scale= 0.1, size=1)
ynew_ms = ynew_m
xnew_ms = xnew_m
```

Now, we are going to run Meanderpy multiple times for the Markov chain Monte Carlo (MCMC) algorithm. Because we need to interpolate the simulated channel for every single iteration, we will define a class that will run meanderpy and interpolate at the same time. 

```ruby
nit = 100                   
depths = D * np.ones((nit,)) 
pad = 0                 
deltas = 50.0               
crdist = 1.8 * W              
kv =  1.0e-12             
dt = 0.1*(365*24*60*60.0)    
dens = 1000               
saved_ts = 1              
n_bends = 5              
Sl = 0.0            
t1 = 0                  
t2 = 0                  
t3 = 0      

def hkm(parm):  
    kl =  (parm[0]*10)/(365*24*60*60.0)  
    Cfs = parm[1] * 0.001 * np.ones((nit,))
    y=cl1[:,0]*10
    x=cl1[:,1]*10
    z=np.zeros(len(x))
    H=depths[0]
    
    try:    
        ch=mp.Channel(-x,-y,z,W,H)
        chb=mp.ChannelBelt(channels=[ch], cutoffs=[], cl_times=[0.0], cutoff_times=[])
        ch = mp.generate_initial_channel(W,D,Sl,deltas,pad,n_bends)
        chb.migrate(nit,saved_ts,deltas,pad,crdist,depths,Cfs,kl,kv,dt,dens,t1,t2,t3,aggr_factor)
        
        if np.shape(0.1*chb.channels[np.int(nit-1)].x)[0] < 1000:
            t = np.linspace(0,1,np.shape(0.1*chb.channels[np.int(nit-1)].x)[0])
            x_m = 0.1*chb.channels[np.int(nit-1)].x
            y_m = 0.1*chb.channels[np.int(nit-1)].y
            fx_m = interpolate.interp1d(t,x_m)
            fy_m = interpolate.interp1d(t, y_m)
            tnew = np.linspace(0,1,1000)
            xnew_m = fx_m(tnew) 
            ynew_m = fy_m(tnew) 
            xnew_m[:] = xnew_m[::-1] 
            ynew_m[:] = ynew_m[::-1] 
        
        else:
            t = np.linspace(0,1,1000)
            xnew_m = np.zeros(1000) 
            ynew_m = np.zeros(1000) 
            ynew_m[:] = ynew_m[::-1] 
            xnew_m[:] = xnew_m[::-1] 

    except:
        t = np.linspace(0,1,1000)
        xnew_m = np.zeros(1000) 
        ynew_m = np.zeros(1000) 
        ynew_m[:] = ynew_m[::-1] 
        xnew_m[:] = xnew_m[::-1] 
        
    return np.array([xnew_m,ynew_m])
```

```ruby
```
