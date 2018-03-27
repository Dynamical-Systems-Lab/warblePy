#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:54:50 2016

@author: alan
"""
import os
import pandas
import numpy
import Wav
import Rec
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot 
import numba
import pdb

class Error(Exception): pass

def finch2(env, fund, Q=None, sigma_amplitude=0.05, sigma_beta=0.002, 
          rate = 44150, steps_per_sample = 20, filter_=True,
          r=0.1,Ch=0.8e-8/360,Lb=1e4,Lg=82,Rb=0.5e7,Rh=6e5,length=4):
    """
    Finch song synthesizer with two sound sources. Based on Sanz Perl et al 2011 and Alonso et al 2016 
    
    env - Wav 
        envelope of the song. Normally calculated with Wav.envelope(method='absint')
    fund - Fund
        temporal evolution of fundamental frecuency.
    Q - Env 
        temporal evolution of the 'detuning parameter'
    sigma_amplitude - float
        range of uniform white noise added to the amplitude
    sigma_beta - float
        Std. Dev. of the normal white noise added to the beta parameter 
    rate - int
        sampling rate of the input and output Wavs
    steps_per_sample - int
        number of integration steps per sample
    filter_ - bool
        indicates if the upper tract filter (UTF), i.e. trachea and oropharyngeal-esophageal cavity (OEC) equations should be applied
    r - float in [0,1]
        reflection coefficient of the trachea with the OEC
    Ch - float
        Equivalent 'capacitance' of the OEC (which works as a Helmholtz resonator)
    Lb - float
        Equivalent 'inductance' of the beak
    Lg - float
        Equivalent 'inductance' of the glottis
    Rb - float
        Equivalent 'resistance' of the beak
    Rh - float
        Equivalent resistance of the OEC (which works as a Helmholtz resonator)
    length - float
        lenght of the trachea in mm
    
    returns a record with components 'env', 'beta', 'fund' and 'syn'
    """
    
    #defining constants
    alfa=-0.15     #alpha is set to -0.15 (negative sign consistent with ODE definition)
    gamma=24000.   #defining gamma time scale constante from eq. 8 Sanz Perl 2011
    c=35000        #speed of sound in mm per second   
    
    
    OEC="OEC.new.dat" #name of the OEC database
    #OEC_PATH = "/home/alan/Lab-LSD/Code/[2016.AB]warblePy/warble_v01/data/OEC.new.dat"
    this_dir, this_filename = os.path.split(__file__)
    OEC_PATH = os.path.join(this_dir, "data", OEC)        
    OEC_db=pandas.read_table(OEC_PATH,header=None)       
    fund2beta=scipy.interpolate.interp1d(OEC_db.ix[:,2], OEC_db.ix[:,1],bounds_error=False
                                   ,fill_value=(OEC_db.iloc[0,1],OEC_db.iloc[-1,1]))
    beta=fund2beta(fund) #calculating beta from fund
    beta=numpy.where(fund<10,0.15*numpy.ones(len(beta)),beta) #set value to 0.15 where fund<10hz
    beta=beta.view(numpy.ndarray)

    if Q is None:
        Q = numpy.ones(len(beta))
    else:
        Q = Q.interpolate(fund.get_times()).view(numpy.ndarray)
        
    beta_L=beta
    beta_R=beta*Q    
    
    #rescaling amplitude to have Q99% = 10, (adequate for noise scale)
    amplitude=env.view(numpy.ndarray).copy()
    amplitude*=10/numpy.percentile(numpy.abs(amplitude),99)    
    
    #integration parameters
    size=min(len(beta),len(amplitude))
    dt=1/(steps_per_sample*rate)    # dt for the integration 
    to=size*steps_per_sample # final index of the time delayed vectors
    
    #initializing sound and labium x position vectors
    sound=numpy.zeros(size+1) #output 
    labium_L=numpy.zeros(size+1)
    labium_R=numpy.zeros(size+1)
    
    #creating noise vectors
    noise_beta_L=numpy.random.randn(to+1)*sigma_beta #normal noise for beta
    noise_beta_R=numpy.random.randn(to+1)*sigma_beta #normal noise for beta
    noise_amplitude=(2*numpy.random.rand(to+1)-1)*sigma_amplitude #uniform noise for amplitude  
     
    if filter_: #complete model with trachea and OEC filter
        v=numpy.zeros(7) #Initial conditions

        #defining arrays to keep tract of preassure inside the different segments of the trachea
        a=numpy.zeros(to+1)  #source (related to the air flow and the oscillations of the labia)
        db=numpy.zeros(to+1) #backward wave at the end of the tract

        #calculating time it takes the wave to travel through a tract segment
        tau=int(length*rate*steps_per_sample/c) #note that tau has index units (for time delay vectors)

        @numba.jit("f8[:](f8[:],f8[:])")
        def takens(v, p): 
            #input vector and parameters
            xL=v[0] # position of the labia (Left)
            yL=v[1] # velocity of the labia (Left)
            xR=v[2] # position of the labia (Right)  
            yR=v[3] # velocity of the labia (Right)             
            i1=v[4] # current i1, i2 and i3 of the equivalent electric circuit
            i2=v[5] # describing the linear filters formed by the trachear and 
            i3=v[6] # the oropharyngeal cavity 

            betaL=p[0] #setting current beta (left)
            betaR=p[1] #setting current beta (Right)
            Vext=p[2] #Transmitted wave at the end of the trachea 
            dVext_dt=p[3] #time derivative of transmitted wave     

            #Takens-Bogdanov normal form. Equation 8 of Sanz Perl et al 2011
            dxL=yL #obs: note that alpha and beta have oposite sign as respect to equation 1 of Boari et al 2015
            dyL=alfa*(gamma**2)+betaL*(gamma**2)*xL-(gamma**2)*(xL**3)-gamma*(xL**2)*yL+(gamma**2)*(xL**2)-gamma*xL*yL

            dxR=yR 
            dyR=alfa*(gamma**2)+betaR*(gamma**2)*xR-(gamma**2)*(xR**3)-gamma*(xR**2)*yR+(gamma**2)*(xR**2)-gamma*xR*yR

            #Equations for the linear filters of the trachear and the oropharyngeal cavity. 
            #Equation 7 of Sanz Perl et al 2011
            di1=i2 #i2 is called omega1 in the Sanz Perl et al 2011 
            di2=-i1/(Lg*Ch) - Rh*(1/Lb+1/Lg)*i2 + i3*(1/(Lg*Ch)-Rb*Rh/(Lb*Lg)) + dVext_dt/Lg + Rh*Vext/(Lg*Lb);
            di3=-Lg*i2/Lb - Rb*i3/Lb + Vext/Lb  
        
            return numpy.array([dxL,dyL,dxR,dyR,di1,di2,di3])
        
        rk4 = make_rk4(takens)
        
        @numba.jit("void(f8[:],f8[:],f8[:],f8[:],f8[:],f8[:])" ,nopython=True)    
        def integrate(v,sound,labium_L,labium_R,a,db):    
            for i in range(int((tau+1)/steps_per_sample), size): #start integrating at tau
                sound[i]=Rb*v[6] #output is i3 (v[6]) times the scaling factor R_b 
                labium_L[i]=v[0] #x 
                labium_R[i]=v[2] #x 
                for j in range(steps_per_sample):
                    i_to=steps_per_sample*i+j #iterator for vectors of length 'to'
                    yL=v[1]
                    yR=v[3]

                    #version modificada por AB al hacer la transicion a Python
                    #2017.03.27 agrego dos fuentes sumandolas 
                    a[i_to] = (.5) * (amplitude[i] + noise_amplitude[i_to]) * (yL + yR) - r * db[i_to-tau] 
                    db[i_to] = a[i_to-tau] # forward wave at the end of the trachea 

                    #The forcer of the Helmholtz resonator (OEC filter) is db 
                    Vext = (1-r)*db[i_to] #Transmitted wave at the end of the trachea 
                    dVext_dt = (1-r)*(db[i_to]-db[i_to-1])*rate*steps_per_sample #time deriv. of trans. wave
                    betaL = beta_L[i] + noise_beta_L[i_to] #setting current beta   
                    betaR = beta_R[i] + noise_beta_R[i_to] #setting current beta   
  
                    p = numpy.array([betaL, betaR, Vext, dVext_dt]) #setting current parameters
                    v=rk4(v,dt,p)

        integrate(v,sound,labium_L,labium_R,a,db) 
       
    else: #simplified model with no filters
        v=numpy.zeros(4) #Initial conditions

        @numba.jit("f8[:](f8[:],f8[:])")
        def takens(v, p): 
            #input vector and parameters
            xL=v[0] # position of the labia (Left) 
            yL=v[1] # velocity of the labia (Left)
            xR=v[2] # position of the labia (Right)
            yR=v[3] # velocity of the labia (Right)
            betaL=p[0] 
            betaR=p[1] 
    
            #Takens-Bogdanov normal form. Equation 8 of Sanz Perl et al 2011
            dxL=yL
            dyL=alfa*(gamma**2)+betaL*(gamma**2)*xL-(gamma**2)*(xL**3)-gamma*(xL**2)*yL+(gamma**2)*(xL**2)-gamma*xL*yL
            dxR=yR
            dyR=alfa*(gamma**2)+betaR*(gamma**2)*xR-(gamma**2)*(xR**3)-gamma*(xR**2)*yR+(gamma**2)*(xR**2)-gamma*xR*yR
        
            return numpy.array([dxL,dyL,dxR,dyR])
        
        rk4 = make_rk4(takens)
        
        @numba.jit("void(f8[:],f8[:],f8[:],f8[:])" ,nopython=True)    
        def integrate(v,sound,labium_L,labium_R):    
            for i in range(size):
                yL=v[1]
                yR=v[3]
                sound[i] = (amplitude[i] + noise_amplitude[i]) * (yL + yR)

                for j in range(steps_per_sample):
                    betaL = beta_L[i] + noise_beta_L[steps_per_sample*i+j]
                    betaR = beta_R[i] + noise_beta_R[steps_per_sample*i+j]
                    p = numpy.array([betaL, betaR]) #setting current beta 
                    v=rk4(v,dt,p)

        integrate(v,sound,labium_L,labium_R) 
        
    sound=sound/numpy.max(numpy.abs(sound))
    #labium=labium/numpy.max(numpy.abs(labium))    

    channels=dict()
    channels['env']= Wav.Wav(env,rate)
    channels['beta']= Wav.Wav(beta,rate)
    channels['fund']= Wav.Wav(fund,rate)
    channels['syn']= Wav.Wav(sound,rate)
    channels['labium_L']= Wav.Wav(labium_L,rate)
    channels['labium_R']= Wav.Wav(labium_R,rate)
    
    return Rec.Rec(**channels)
    


def finch(env, beta=None, fund=None, sigma_amplitude=0.05, sigma_beta=0.002, 
          rate = 44150, steps_per_sample = 20, filter_=True,
          r=0.1,Ch=0.8e-8/360,Lb=1e4,Lg=82,Rb=0.5e7,Rh=6e5,length=4):
    """
    Finch song synthesizer. Based on Sanz Perl et al 2011. 
    
    env - Wav 
        envelope of the song. Normally calculated with Wav.envelope(method='absint')
    beta - Env 
        temporal evolution of the beta parameter. Not required if 'fund' is given
    fund - Fund
        temporal evolution of fundamental frecuency. Not required if 'beta' is given
    sigma_amplitude - float
        range of uniform white noise added to the amplitude
    sigma_beta - float
        Std. Dev. of the normal white noise added to the beta parameter 
    rate - int
        sampling rate of the input and output Wavs
    steps_per_sample - int
        number of integration steps per sample
    filter_ - bool
        indicates if the upper tract filter (UTF), i.e. trachea and oropharyngeal-esophageal cavity (OEC) equations should be applied
    r - float in [0,1]
        reflection coefficient of the trachea with the OEC
    Ch - float
        Equivalent 'capacitance' of the OEC (which works as a Helmholtz resonator)
    Lb - float
        Equivalent 'inductance' of the beak
    Lg - float
        Equivalent 'inductance' of the glottis
    Rb - float
        Equivalent 'resistance' of the beak
    Rh - float
        Equivalent resistance of the OEC (which works as a Helmholtz resonator)
    length - float
        lenght of the trachea in mm
    
    returns a record with components 'env', 'beta', 'fund' and 'syn'
    """
    
    #defining constants
    alfa=-0.15     #alpha is set to -0.15 (negative sign consistent with ODE definition)
    gamma=24000.   #defining gamma time scale constante from eq. 8 Sanz Perl 2011
    c=35000        #speed of sound in mm per second   
    
    if not((beta is not None) or (fund is not None)):
        raise Error("beta or fund required")
    
    if beta is None: #calculating beta from fund based on OEC database
        OEC="OEC.new.dat" #name of the OEC database
        #OEC_PATH = "/mnt/Storage/git/warblePy/data/OEC.new.dat"
        this_dir, this_filename = os.path.split(__file__)
        OEC_PATH = os.path.join(this_dir, "data", OEC)        
        OEC_db=pandas.read_table(OEC_PATH,header=None)       
        fund2beta=scipy.interpolate.interp1d(OEC_db.iloc[:,2], OEC_db.iloc[:,1],bounds_error=False
                                   ,fill_value=(OEC_db.iloc[0,1],OEC_db.iloc[-1,1]))
        beta=fund2beta(fund) #calculating beta from fund
        beta=numpy.where(fund<10,0.15*numpy.ones(len(beta)),beta) #set value to 0.15 where fund<10hz
    
    beta=beta.view(numpy.ndarray)
    amplitude=env.view(numpy.ndarray).copy()
    
    #rescaling amplitude to have Q99% = 10, (adequate for noise scale)
    amplitude*=10/numpy.percentile(numpy.abs(amplitude),99)    
    
    #integration parameters
    size=min(len(beta),len(amplitude))
    dt=1/(steps_per_sample*rate)    # dt for the integration 
    to=size*steps_per_sample # final index of the time delayed vectors
    
    #initializing sound and labium x position vectors
    sound=numpy.zeros(size+1) #output 
    labium=numpy.zeros(size+1)
    
    #creating noise vectors
    noise_beta=numpy.random.randn(to+1)*sigma_beta #normal noise for beta
    noise_amplitude=(2*numpy.random.rand(to+1)-1)*sigma_amplitude #uniform noise for amplitude  
     
    if filter_: #complete model with trachea and OEC filter
        v=numpy.zeros(5) #Initial conditions

        #defining arrays to keep tract of preassure inside the different segments of the trachea
        a=numpy.zeros(to+1)  #source (related to the air flow and the oscillations of the labia)
        db=numpy.zeros(to+1) #backward wave at the end of the tract

        #calculating time it takes the wave to travel through a tract segment
        tau=int(length*rate*steps_per_sample/c) #note that tau has index units (for time delay vectors)

        @numba.jit("f8[:](f8[:],f8[:])")
        def takens(v, p): 
            #input vector and parameters
            x=v[0] # position of the labia  
            y=v[1] # velocity of the labia
            i1=v[2] # current i1, i2 and i3 of the equivalent electric circuit
            i2=v[3] # describing the linear filters formed by the trachear and 
            i3=v[4] # the oropharyngeal cavity           
            beta1=p[0] #setting current beta  
            Vext=p[1] #Transmitted wave at the end of the trachea 
            dVext_dt=p[2] #time derivative of transmitted wave     

            #Takens-Bogdanov normal form. Equation 8 of Sanz Perl et al 2011
            dx=y #obs: note that alpha and beta have oposite sign as respect to equation 1 of Boari et al 2015
            dy=alfa*(gamma**2)+beta1*(gamma**2)*x-(gamma**2)*(x**3)-gamma*(x**2)*y+(gamma**2)*(x**2)-gamma*x*y
            #Equations for the linear filters of the trachear and the oropharyngeal cavity. 
            #Equation 7 of Sanz Perl et al 2011
            di1=i2 #i2 is called omega1 in the Sanz Perl et al 2011 
            di2=-i1/(Lg*Ch) - Rh*(1/Lb+1/Lg)*i2 + i3*(1/(Lg*Ch)-Rb*Rh/(Lb*Lg)) + dVext_dt/Lg + Rh*Vext/(Lg*Lb);
            di3=-Lg*i2/Lb - Rb*i3/Lb + Vext/Lb  
        
            return numpy.array([dx,dy,di1,di2,di3])
        
        rk4 = make_rk4(takens)
        
        @numba.jit("void(f8[:],f8[:],f8[:],f8[:],f8[:])" ,nopython=True)    
        def integrate(v,sound,labium,a,db):    
            for i in range(int((tau+1)/steps_per_sample), size): #start integrating at tau
                sound[i]=Rb*v[4] #output is i3 (v[4]) times the scaling factor R_b 
                labium[i]=v[0] #x 
                for j in range(steps_per_sample):
                    i_to=steps_per_sample*i+j #iterator for vectors of length 'to'
                    y=v[1]

                    #version modificada por AB al hacer la transicion a Python
                    a[i_to] = (.5) * (amplitude[i] + noise_amplitude[i_to]) * y - r * db[i_to-tau] 
                    db[i_to] = a[i_to-tau] # forward wave at the end of the trachea 

                    #The forcer of the Helmholtz resonator (OEC filter) is db 
                    Vext = (1-r)*db[i_to] #Transmitted wave at the end of the trachea 
                    dVext_dt = (1-r)*(db[i_to]-db[i_to-1])*rate*steps_per_sample #time deriv. of trans. wave
                    beta1 = beta[i] + noise_beta[i_to] #setting current beta    
  
                    p = numpy.array([beta1, Vext, dVext_dt]) #setting current parameters
                    v=rk4(v,dt,p)

        integrate(v,sound,labium,a,db) 
       
    else: #simplified model with no filters
        v=numpy.zeros(2) #Initial conditions

        @numba.jit("f8[:](f8[:],f8[:])")
        def takens(v, p): 
            #input vector and parameters
            x=v[0] # position of the labia  
            y=v[1] # velocity of the labia
            beta1=p[0] 
    
            #Takens-Bogdanov normal form. Equation 8 of Sanz Perl et al 2011
            dx=y
            dy=alfa*(gamma**2)+beta1*(gamma**2)*x-(gamma**2)*(x**3)-gamma*(x**2)*y+(gamma**2)*(x**2)-gamma*x*y
        
            return numpy.array([dx,dy])
        
        rk4 = make_rk4(takens)
        
        @numba.jit("void(f8[:],f8[:],f8[:])" ,nopython=True)    
        def integrate(v,sound,labium):    
            for i in range(size):
                y=v[1]
                sound[i] = (amplitude[i] + noise_amplitude[i]) * y

                for j in range(steps_per_sample):
                    p = numpy.array([beta[i] + noise_beta[steps_per_sample*i+j]]) #setting current beta 
                    v=rk4(v,dt,p)

        integrate(v,sound,labium) 
        
    maxabs_sound = numpy.nanmax(numpy.abs(sound))    
    if not numpy.isnan(maxabs_sound) and maxabs_sound>0:
        sound=sound/numpy.max(numpy.abs(sound))
    
    #labium=labium/numpy.max(numpy.abs(labium))    

    channels=dict()
    channels['env']= Wav.Wav(env,rate)
    channels['beta']= Wav.Wav(beta,rate)
    channels['fund']= Wav.Wav(fund,rate)
    channels['syn']= Wav.Wav(sound,rate)
    channels['labium']= Wav.Wav(labium,rate)
    
    return Rec.Rec(**channels)
    
    
def make_rk4(deri):
    """
    Runge-Kutta function creator for a specific autonomous ODE model defined by deriv

    deri - numba function with arguments v (numpy array), and p (numpy array or scalar)
           expected to return a numpy array with the derivatives

    Note that rk4 is compiles each time this function is called  
    
    returns a numba function rk4(h,dt,p)
        h is the numpy array of the model's variables
        dt is the integration step
        p are parameters passed to deri
    """
    @numba.jit("f8[:](f8[:],f8,f8[:])")
    def rk4(h,dt,p):
        dt2=dt/2
        dt6=dt/6

        k1 = deri(h,p)
        k2 = deri(h + dt2*k1,p)
        k3 = deri(h + dt2*k2,p)
        k4 = deri(h + dt*k3,p)  
        
        return h + dt6*(2*(k2+k3)+k1+k4)
    return rk4    

 


def db2fund(FF_db,start=0,end=None,sampling=44150,FF0=0,**kwargs):
    """
    Create a Fund object from a pandas.DataFrame
    
    FF_db - pandas.DataFrame
        DataFrame containing the frecuency information. Columns: 'type', 'syl', 'time', 'freq', 'harmonic'
    start - float
        time in seconds at which the Fund object should start
    end - float
        time in seconds at which the Funs object should end.
    sampling - int
        the sampling rate of the Fund object
    FF0 - float
        frequency of time laps without points in FF_db
    kwargs - dict
        further arguments for add_silence_FF_pts
    """
    assert 'type' in FF_db.columns
    assert 'syl' in FF_db.columns
    assert 'time' in FF_db.columns
    assert 'freq' in FF_db.columns
    assert 'harmonic' in FF_db.columns
    
    FF0_db=add_silence_FF_pts(FF_db,FF0=FF0,**kwargs)
    if start is None: start=min(FF0_db.time)
    if end is None: end=max(FF0_db.time)
    
    linint=scipy.interpolate.interp1d(FF0_db.time, FF0_db.freq/FF0_db.harmonic,bounds_error=False,fill_value=FF0)
    output=Wav.Fund(linint(numpy.arange(start,end,1/sampling)),rate=sampling,delay=start)
    
    kwargs['FF0']=FF0
    kwargs['sampling']=sampling
    output.method="Syn.db2fund"
    output.method_args=kwargs

    return output
    
    
def add_silence_FF_pts(FF_db,FF0=0,delta_t=0.005):
    """
    Adds points to 'silent gaps' in the fundamental frecuency DataFrame
    
    FF_db - pandas.DataFrame
        DataFrame containing the frecuency information. Columns: 'type', 'syl', 'time', 'freq', 'harmonic'
    FF0 - float
        frequency of time laps without points in FF_db
    delta_t - float
        time insterval between consecutive 'silent' points
        
    return a pandas.DataFrame with new rows in the silent gaps
    """

    #cehcking the DataFrame is as expected
    assert 'type' in FF_db.columns
    assert 'syl' in FF_db.columns
    assert 'time' in FF_db.columns
    assert 'freq' in FF_db.columns
    assert 'harmonic' in FF_db.columns
    
    silence_db=pandas.DataFrame()
    t=min(FF_db.time)
    silence_count=1
    for syl in FF_db.syl.unique():
    #syl="i2"
        syl_db=FF_db.ix[FF_db.syl==syl,:]
        syl_t=syl_db.time
        silence_t=numpy.arange(t+delta_t,min(syl_t)-delta_t,delta_t)
        if len(silence_t)>0:
            syl_silence_db=pandas.concat([syl_db.iloc[1]] * len(silence_t), axis=1).transpose()
            syl_silence_db.time=silence_t
            syl_silence_db.freq=FF0
            syl_silence_db.harmonic=1
            syl_silence_db.syl='silence'+str(silence_count)
            syl_silence_db.type='silence'
            silence_db=pandas.concat([silence_db,syl_silence_db],ignore_index=True)
            silence_count+=1
        t=max(syl_t)        
    FF_db=pandas.concat([FF_db,silence_db],ignore_index=True)           
    FF_db=FF_db.sort_values(['time'], axis=0, ascending=True).reset_index(drop=True)
    return FF_db
    
def plot_filter(r=0.1,Ch=0.8e-8/360,Lb=1e4,Lg=82,Rb=0.5e7,Rh=6e5,length=4,rate=44150,numtaps=1001):
    """
    Plots the steady-sate upper tract filter of the model. Based on Sanz Perl et al 2011
    
    r - float in [0,1]
        reflection coefficient of the trachea with the OEC
    Ch - float
        Equivalent 'capacitance' of the OEC (which works as a Helmholtz resonator)
    Lb - float
        Equivalent 'inductance' of the beak
    Lg - float
        Equivalent 'inductance' of the glottis
    Rb - float
        Equivalent 'resistance' of the beak
    Rh - float
        Equivalent resistance of the OEC (which works as a Helmholtz resonator)
    length - float
        lenght of the trachea in mm   
    """
    
    freqT = 35000/(4 * length)    
    tau=1/(4*freqT)
    f=numpy.linspace(0,rate/2,numtaps)
    gamma1=(numpy.exp(1j*2*numpy.pi*f*tau)*(1-r)/((numpy.exp(2*1j*(2*numpy.pi*f)*tau)+r)))
    gamma2=1/(Rb+(1j*(Lb+Lg)*(2*numpy.pi*f))+((Ch*Lg*((2*numpy.pi*f)**2)*(1j*Rb-Lb*(2*numpy.pi*f)))/(Ch*Rh*(2*numpy.pi*f)-1j)))
    matplotlib.pyplot.plot(f,numpy.abs(gamma1*gamma2))  
    
def finch_FIR_filter(r=0.1,Ch=0.8e-8/360,Lb=1e4,Lg=82,Rb=0.5e7,Rh=6e5,length=4,rate=44150,numtaps=1001):
    """
    returns the coefficients of a Finite Impulse Response filter, corresponding to the steady-state Upper Tract Filter of the model
    
    r - float in [0,1]
        reflection coefficient of the trachea with the OEC
    Ch - float
        Equivalent 'capacitance' of the OEC (which works as a Helmholtz resonator)
    Lb - float
        Equivalent 'inductance' of the beak
    Lg - float
        Equivalent 'inductance' of the glottis
    Rb - float
        Equivalent 'resistance' of the beak
    Rh - float
        Equivalent resistance of the OEC (which works as a Helmholtz resonator)
    length - float
        lenght of the trachea in mm  
    """
    freqT = 35000/(4 * length)    
    tau=1/(4*freqT)
    f=numpy.linspace(0,rate/2,numtaps)
    gamma1=(numpy.exp(1j*2*numpy.pi*f*tau)*(1-r)/((numpy.exp(2*1j*(2*numpy.pi*f)*tau)+r)))
    gamma2=1/(Rb+(1j*(Lb+Lg)*(2*numpy.pi*f))+((Ch*Lg*((2*numpy.pi*f)**2)*(1j*Rb-Lb*(2*numpy.pi*f)))/(Ch*Rh*(2*numpy.pi*f)-1j)))
    return scipy.signal.firwin2(numtaps,f,numpy.abs(gamma1*gamma2),nyq=rate/2)

        
        