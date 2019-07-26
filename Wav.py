#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wav.py module of WarblePy. This module defines the Wav, Env, Cor and Warped object and methods.

Wav: represents a wave object, containing sound or EMG activity. These objects 
     can represent .wav files.  
Env: Inherits from Wav. Represents the 'envelope' of a Wav object
Cor: Inherits from Wav. Represents the correlation between two Wav objects
Warped: Inherits from Wav. Represents a time warped version of a Wav. 
---
Author: Alan Bush (loosely inspired in wavio by Warren Weckesser)
"""

import os
import scipy.io.wavfile 
import scipy.signal
import scipy.interpolate
import scipy.optimize
import numpy
import math
import matplotlib.pyplot 
import matplotlib.mlab
import pandas
import tempfile
import warnings
import numba
import collections
import pdb


class Error(Exception): pass

class Wav(numpy.ndarray):
    """
    Object containing the data of or wav file (or equivalent).
    
    Inherits from numpy.ndarray. Additional attributes/methods are:
        
    rate: float
        The sample rate of the WAV file.        
    duration: float
        The duration of the WAV in seconds
    delay: float
        The delay until the start of the data (relative to other Wav from Record)
    tjust: float between in [0,1]
        Time justification. Were, in the segment it represents, a sample is defined
        tjust=1 implies it is at the end, tjust=0 at the beggining.
    samples: int
        Number of samples in the WAV file
    n_channels: int
        Number of channels in the WAV file
    env: Env or None
        envelope of the WAV file calculated by self.envelope(). Defaults to None.
    spe: Spe or None
        spectrogram of the WAV file calculated by seld.spectrogram(). Defaults to None. 
    """
    def __new__(cls, input_array, rate, delay=0, channel="", tjust=0.5):
        if input_array.ndim > 2: Error("input_array must have ndim <=2")
        obj = numpy.asarray(input_array).view(cls)
        obj.rate = float(rate)
        obj.delay = float(delay)
        obj.tjust=float(tjust)
        obj.env = None
        obj.spe = None
        obj.file_name=""
        obj.file_path=""
        obj.channel=channel
        obj.method = None
        obj.method_args = None
        return obj
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self.rate = getattr(obj, 'rate', None)
        self.delay = getattr(obj, 'delay', 0)
        self.tjust = getattr(obj, 'tjust', 0)
        self.env = getattr(obj, 'env', None)
        self.spe = getattr(obj, 'spe', None)
        self.file_name = getattr(obj, 'file_name', '')
        self.file_path = getattr(obj, 'file_path', '')
        self.channel = getattr(obj, 'channel', '')
        self.method = getattr(obj, 'method', None)
        self.method_args = getattr(obj, 'method_args', None)

    def __repr__(self):
        if self.ndim == 1:
            s = ("%s(rate=%dHz, duration=%0.3fsec, samples=%i, channel=%s, delay=%0.3fsec, tjust=%0.1f)" %
                 (self.__class__.__name__, self.rate, self.duration, self.samples, self.channel, self.delay, self.tjust))
            if len(self.file_name)>0: s=s+"\n"+self.file_name
        elif self.ndim == 0:
            s = numpy.float64(self).__repr__()
        else:
            s = super(Wav,self).__repr__()
        return s

    def __len__(self):
        return self.shape[0]
        
    def copy(self,*args,**kwargs):
        c=type(self)(self.array.copy(*args,**kwargs),self.rate,delay=self.delay,\
               channel=self.channel,tjust=self.tjust)
        c.env = self.env
        c.spe = self.spe
        c.file_name=self.file_name
        c.file_path=self.file_path
        c.channel=self.channel
        c.method = self.method
        if self.method_args is not None:
            c.method_args = self.method_args.copy()
        return c

    @property 
    def array(self):
        """
        return the underlaying numpy vector
        """
        return self.view(numpy.ndarray)
        
    @property
    def start(self):
        """
        The start of the time being represented by the Wav object
        """
        return self.delay 

    @property
    def end(self):
        """
        End time of the sinal represented by the Wav
        """
        return self.delay + self.duration 

    @property
    def tlim(self):
        """
        2-tuple of start and end times for the Wav
        """
        return(self.start,self.end)        
        
    @property
    def sampling(self):
        return self.rate
        
    @property
    def samples(self):
        """
        Number of samples of the Wav
        """
        return self.shape[0]
        
    @property
    def duration(self):
        """
        Duration of the sinal represented by the Wav
        """
        return float(self.samples/self.rate) #float((self.samples-1)/self.rate)

    @classmethod
    def read(cls, file, delay=None, norm_max=None, norm_min=None, norm_factor=None,\
             norm=None, tjust=None, channel=None, log="_warble-normalization.dat"):
        """
        Opens a WAV file and returns a Wav object

        Parameters
        ----------
        file - string or open file handle
            Input wav file.
        delay - float
            delay in seconds relative to other Wavs of the record. 
            defaults to log value or 0.0
        norm_max - float
            max value used in the normalization
            defaults to log value or 1.0
        norm_min - float
            min value used in the normalization
            defaults to log value or -1.0
        norm_factor - float
            normalization factor used in the normalization
            defaults to log value or 0.999
        norm - bool
            if True applies Wav = (Wav/2*norm_factor)(norm_max-norm_min)+(norm_max+norm_min)/2
            defaults to log True if norm_max, norm_min and norm_factor are valid
        tjust - float in [0,1]
            were is the sample time point given in the segment it represets
            defaults to log value or 0.5
        channel - str
            channel description
            defaults to log value or ''
        log - str
            name of the log file where to look fornormalization and delay values
            if not found hardcoded defaults are used
            
        returns a Wav object
        """
        
        #using log values if not provided in call
        if log is not None: 
            folder=os.path.dirname(file)
            file_name=os.path.basename(file)
            log_fname = os.path.join(folder,log)
            if os.path.isfile(log_fname):
                log_db=pandas.read_table(log_fname)
                if any(log_db.file==file_name):
                    row=log_db.ix[log_db.file==file_name]
                    if norm_max is None:
                        norm_max=float(row.norm_max.iloc[0])
                    if norm_min is None:
                        norm_min=float(row.norm_min.iloc[0])
                    if norm_factor is None:
                        norm_factor=float(row.norm_factor.iloc[0])
                    if delay is None:
                        delay=float(row.delay.iloc[0])
                    if tjust is None:
                        tjust=float(row.tjust.iloc[0])
                    if channel is None:
                        channel=str(row.channel.iloc[0])  

        #setting default values if not provided and not in log
        if norm_max is None:
            norm_max=1.0
        if norm_min is None:
            norm_min=-1.0
        if norm_factor is None:
            norm_factor=0.999
        if delay is None:
            delay=0.0
        if tjust is None:
            tjust=0.5
        if channel is None:
            channel=''
        if norm is None:
            norm = not numpy.isnan(norm_max) and not numpy.isnan(norm_min) and not numpy.isnan(norm_factor)
        
        rate, input_array = scipy.io.wavfile.read(file)
        if input_array.dtype.name=="int16": 
            nbits=16 
        else:
            nbits=None
        input_array=input_array.astype(numpy.float64,order='C',casting='safe',subok=False,copy=True)
        if norm is True: 
            if nbits is None: raise Error("Can only read int16 wav files")
            input_array/=2**(nbits-1) #normalization by the max possible number given de bit depth
            input_array*=(norm_max-norm_min)/(2*norm_factor) #denormalizing to original values
            input_array+=(norm_max+norm_min)/2 #centering in zero
        obj=cls.__new__(cls,input_array,rate=rate,delay=delay,tjust=tjust,channel=channel)
        obj.file_name=os.path.basename(file)
        obj.file_path=os.path.dirname(file)
        return obj
        
    @classmethod     
    def zeros(cls,template=None,rate=None,duration=None,samples=None,delay=None,tjust=None):
        """
        Creates a Wav object filled with zeros

        template - Wav
            object from which to copy the rate and delay
        rate - float
            sampling rate to define time vector. Used if template is None.
            if used, either samples or duration should be given as well
        duration - float
            duration in seconds. Used if template is None.
            if used, either samples or rate should be given as well
        samples - integer
            number of time samples. Used if template is None.
            if used, either duration or rate should be given as well
        delay - float or None
            time delay in seconds
            defaults to template.delay or 0 if None
        tjust: float in [0,1] or None
            were is the sample time point given in the segment it represets
            defaults to template.tjust or 0.5 if None
        """

        assert template is not None or ((rate is not None) + (duration is not None) + (samples is not None) == 2)
        
        if template is not None:
            samples=template.samples
            rate=template.rate
            if delay is None:
                delay=template.delay
            if tjust is None:
                tjust=template.tjust
        else:
            if rate is None:
                rate = samples/duration
            if samples is None:
                samples=max([1,int(duration*rate)])
            if delay is None:
                delay=0
            if tjust is None:
                tjust=0.5
                
        return cls(numpy.zeros(samples),rate=rate,delay=delay,tjust=tjust)
        
    @classmethod
    def create(cls, times, values, rate=44150, tjust=0.5, channel=""):
        """
        Creates a wav object from 'times' and 'values' vectors
        
        times - ndarray
            numpy vector of times at which the values were mesured
        values - ndarray
            numpy vector of values
        rate - float/int
            sampling rate of the output Wav
        tjust - float in [0,1]
            were is the sample time point given in the segment it represets
         channel - str
            channel description
            
        return a Wav object
        """
        linint = scipy.interpolate.interp1d(times,values,bounds_error=False,fill_value='extrapolate')
        delay = numpy.amin(times)
        samples = int((numpy.amax(times) - delay)*rate)
        duration = samples/rate
        wav_times=numpy.linspace(delay+tjust/rate,delay+duration+tjust/rate,num=samples,endpoint=False)
        wav_values=linint(wav_times)
        return cls(wav_values,rate=rate,delay=delay,tjust=tjust,channel=channel)
        
        
    def write(self, file, format_='int16', normalize=True, norm_factor=0.999, log="_warble-normalization.dat"):
        """
        Writes the Wav object to a wav file
        
        file - str
            Output wav file.
        format_ - str
            Either float32 or int16
        normalize - bool
            should the Wav be normalized
        log - str
            name of the log file used to save the normalization value
            if None the normalizatin value is not saved
        """
        
        #import pdb; pdb.set_trace()
        
        array = self.view(numpy.ndarray)

        if normalize: 
            maxabs=numpy.amax(numpy.abs(array))
            array = array*norm_factor/maxabs
        else:
            maxabs = numpy.nan
        
        if log is not None:
            folder=os.path.dirname(file)
            fname=os.path.basename(file)
            log_fname=os.path.join(folder,log)

            if os.path.isfile(log_fname):
                log_db=pandas.read_table(log_fname)
                if 'delay' not in log_db.columns:
                    log_db=log_db.assign(delay=0)    
                if 'tjust' not in log_db.columns:
                    log_db=log_db.assign(tjust=0.5)
                if 'channel' not in log_db.columns:
                    log_db=log_db.assign(channel='')
            else:
                log_db=pandas.DataFrame(columns=['file','norm_max','norm_min','norm_factor','delay','tjust','channel']) 

            if any(log_db.file==fname):
                log_db.ix[log_db.file==fname,['norm_max','norm_min','norm_factor','delay','tjust','channel']]=\
                    [maxabs,-maxabs,norm_factor,self.delay,self.tjust,self.channel]
            else:
                log_db.loc[len(log_db)]=[fname,maxabs,-maxabs,norm_factor,self.delay,self.tjust,self.channel]
        
            log_db.to_csv(log_fname,sep='\t',index=False,na_rep='NA')

        if format_=='float32':
            scipy.io.wavfile.write(file,int(self.rate),array.astype(numpy.float32))
        elif format_ == 'int16':
            max_value = numpy.amax(numpy.abs(array)) 
            if max_value <= 1: array*=32766
            scipy.io.wavfile.write(file,int(self.rate),array.view(numpy.ndarray).astype(numpy.int16))
        else:
            raise Error("unknown format_")
        
        #updating name and path attributes    
        self.file_name=os.path.basename(file)
        self.file_path=os.path.dirname(file)
        
        
    def to_dataframe(self,channel=None,**kwargs):
        """
        Returns a data frame representation of the object
        
        channel - string
            name of the channel. Defaults to self.channel if defined
        kwargs - dict
            columns to be appended to the dataframe
            
        returns a pandas.DataFrame
        """
        if channel is None:
            if self.channel is None or self.channel=='':
                channel=str(type(self).__name__)
            else:
                channel=self.channel

        df = pandas.DataFrame({'time':self.get_times()})
        df[channel]=self.array
        for col in kwargs:
            df[col]=kwargs[col]
        return df
        
    def reverse(self):
        """
        reverse the Wav
        """
        return Wav(self[::-1],self.rate, delay=self.delay, channel=self.channel, tjust=self.tjust)
        
        
    def normalize(self,method='absmax',alpha=1,beta=1):
        """
        normalize the Wav intensity
        
        method - str
            method used for normalization: 'absmax', 'meanstd', 'alphabeta'
        alpha - float
            used for alphabeta method
        beta - float
            used for alphabeta method

        returns a normalized wav object
        """
        if method == 'absmax':
            norm_fact=numpy.max(numpy.abs(self))
            return Wav(self/norm_fact,self.rate, delay=self.delay, channel=self.channel, tjust=self.tjust)
        elif method == 'meanstd':
            return (self-numpy.nanmean(self.array))/numpy.nanstd(self.array)
        elif method == 'alphabeta':
            return (self-alpha*numpy.nanmean(self.array))/numpy.power(numpy.nanstd(self.array),beta)
        else:
            raise Error("unknown normalization method")
            
    def clip(self,min_,max_):
        """
        Clips the wav object
        """
        return type(self)(numpy.clip(self.view(numpy.ndarray), min_, max_), 
                       self.rate, delay=self.delay, channel=self.channel)
        
    def crop(self,start=None,end=None,samples=None,aligned=True):
        """
        Returns a cropped copy of the object
        
        start: float
            start time in seconds
        end: float
            end time in seconds
        samples: integer
            number of samples to return
        aligned, bool
            indicates if cropped Wav should be time aligned with original Wav.
            if False delay is set to zero. 
        """
        if start is not None: start=float(start)
        if end is not None: end=float(end)
        crop_idx = self._crop_idx(start,end,samples)
        if crop_idx is None: return None
        (start_idx, end_idx) = crop_idx
        output = self[start_idx:end_idx]
        #output.delay = float(max([self.delay, self.get_time(start_idx)])*bool(aligned))
        output.delay = float((self.get_time(start_idx)-self.tjust/self.rate)*bool(aligned))
        output.tjust=self.tjust
        output.file_path=''
        output.file_name=''
        return output
        
    def _crop_idx(self,start=None,end=None,samples=None):    
        """
        Internal function to calculate crop indexes
        If interval required is not contained in Wav, returns None
        """
        assert (start is not None) + (end is not None) + (samples is not None) == 2
        if start is None:
            if samples is None:
                start=0
                start_idx=0
        else:
            start_idx=math.floor((start-self.delay)*self.rate)
            if start_idx <= 0: start_idx = 0
            if start_idx >= len(self): return None

        if end is None:
            if samples is None:
                end=self.duration
                end_idx=len(self)
            else:
                end_idx = start_idx + samples 
                if end_idx < 0: end_idx = 0
                if end_idx >= len(self): end_idx = len(self)
        else:
            end_idx=math.floor((end-self.delay)*self.rate)
            if end_idx <= 0: return None
            if end_idx >= len(self): end_idx = len(self)

        if start is None and samples is not None:
            start_idx = end_idx - samples
            if start_idx < 0: start_idx = 0
            if start_idx >= len(self): return None
     
        assert start_idx < end_idx    
        return (start_idx, end_idx)

        
    def get_times(self, *args, **kwargs):
        """
        returns vector of times for each data point
        
        index - numpy integer vector of indexes of times of interest. 
                If None, times for all data points are returned
        """
        return self.get_time(*args, **kwargs)

    def get_time(self, index=None):
        """
        returns vector of times for each data point
        
        index - numpy integer vector of indexes of times of interest. 
                If None, times for all data points are returned
        """
        if index is None:
            return numpy.linspace(self.delay+self.tjust/self.rate,self.delay+self.duration+self.tjust/self.rate,num=self.samples,endpoint=False)
        else:
            return self.delay + self.duration*index/self.samples + self.tjust/self.rate

    def get_values(self, *args, **kwargs):
        """
        returns an array with the values interpolated at the given times
        """
        return self.get_value(*args, **kwargs)

    def get_value(self, time):
        """
        returns an array with the values interpolated at the given times
        """
        #using linear interpolation
        linint=scipy.interpolate.interp1d(self.get_time(), self.array, bounds_error=False, fill_value=0)
        return linint(time)
            
    def time_max(self):
        """
        returns time of max as calculated by numpy.argmax
        """
        if numpy.all(numpy.isnan(self)):
            return numpy.nan
        else:
            return self.get_time(numpy.nanargmax(self))

    def time_min(self):
        """
        returns time of min as calculated by numpy.argmax
        """
        if numpy.all(numpy.isnan(self)):
            return numpy.nan
        else:
            return self.get_time(numpy.nanargmin(self))
        
    def shift(self,delay=None,start=None,end=None,by=None):
        """
        returns a time shifted copy of the wav
        
        delay - float
            new delay
        start - float
            new start
        end - float
            new end
        by - float 
            time in seconds by which to shif the wav relative to current time
        """        
        Nargs=(delay is not None)+(start is not None)+(end is not None)+(by is not None)
        if Nargs > 1:
            raise Error("only one argument should be passed to shift")
        if Nargs == 0:
            raise Error("argument required for shift")
        s=self.copy()
        if start is not None:
            delay = start
        if delay is not None:
            s.delay=delay
            return s
        if end is not None:
            s.delay = end - s.duration            
            return s
        if by is not None:
            s.delay = s.delay + by
            return s
            
    def set_delay(self,delay):
        """
        sets the delay of the Wav
        """
        #warnings.warn("set_delay/set_start deprecated. Use 'shift' instead")
        if delay is not None:
            self.delay=delay
        return self
        
    def set_start(self,start):
        """
        sets the start time of the Wav
        """
        return self.set_delay(start)
        
    def apply(self,method,_rate=None,_delay=None,_tjust=None,**kwargs):    
        """
        Apply a method to the wav
        
        mehod - callable
            function to be applied to self.array
    
        _rate - float
            sampling rate of the resulting Env. If None self.rate is used
            
        _delay - float
            delay of the resulting Env. If None self.delay is used
            
        _tjust - float
            tjust of the resulting Env. If None self.tjust is used
            
        kwargs - dict
            further arguments for method
            
        returns a Env
        """
        if _rate is None:
            _rate = self.rate
        if _delay is None:
            _delay = self.delay
        if _tjust is None:
            _tjust = self.tjust
        
        input_array=method(self.array,**kwargs)
        a=Env(input_array,rate=_rate,delay=_delay,tjust=_tjust)       
        a.method="apply"
        a.method_args=kwargs
        return a        
        
    def derivative(self,method='gradient'):
        """
        Calculate the derivative of the Wav
        
        method - str
            method to be used. Only 'gradient' is currently implemented.
        """
        if method!='gradient':
            raise Error("Ony method gradient is implemented for derivative")
        
        input_array=numpy.gradient(self.array,1/self.rate)
        return Env(input_array,rate=self.rate,delay=self.delay,tjust=self.tjust)   
    
    def smooth(self, method="savgol", **kwargs):
        """
        Calculates a smooth version of the Wav
        """
        known_methods=['savgol']
        method=method.lower()
        method_name="_smooth_" + method
        if not method in known_methods:
            raise Error("Method "+method+" not implemented")
        smooth_method=getattr(self,method_name)
        s=smooth_method(**kwargs)
        s.method=method
        s.method_args=kwargs
        s.channel=self.channel + "_smooth"
        return s
    
    def _smooth_savgol(self, window=7, order=3, **kwargs):
        """
        Apply a Savitzky-Golay filter
        
        window - int
            The length of the filter window (i.e. the number of coefficients). window must be a positive odd integer.
            
        order - int
            The order of the polynomial used to fit the samples. order must be less than window.
            
        kwargs - dict
            further arguments for scipy.signal.savgol_filter
            
        returns a Env with the smoothen data
        """
        input_array=scipy.signal.savgol_filter(self.array, window, order, **kwargs)
        return Env(input_array,rate=self.rate,delay=self.delay,tjust=self.tjust)          
        
    def envelope(self, method="jarne", **kwargs):
        """
        Calculatethe envelope of the Wav with the indicated method
        """
        known_methods=['binabs','binmax','binmin','binmap','jarne','mindlin','absint','robust','teager','robust_teager','log_robust_teager']
        method=method.lower()
        method_name="_envelope_" + method
        if not method in known_methods:
            raise Error("Method "+method+" not implemented")
        envelope_method=getattr(self,method_name)
        self.env=envelope_method(**kwargs)
        self.env.method=method
        self.env.method_args=kwargs
        self.env.channel=self.channel + " envelope"
        return self.env
        
    def _envelope_binmap(self, func=None, bin_size=35, repeat=1):
        """
        Calculate func over bins of 'bin_size' points, repeating each value 'repeat' times
        """
        input_array=numpy.array(list(map(func,numpy.array_split(self,math.floor(self.samples/bin_size))))).repeat(repeat)
        rate=self.rate*(repeat/bin_size)
        return Env(input_array,rate=rate,delay=self.delay,tjust=self.tjust)  
        
    def _envelope_binmax(self, bin_size=35, repeat=1):
        """
        Calculate envelope as max value in bins of 'bin_size' points, repeating each value 'repeat' times
        Note that the abs is not calculated prior to max.
        """
        input_array=numpy.array(list(map(max,numpy.array_split(self,math.floor(self.samples/bin_size))))).repeat(repeat)
        rate=self.rate*(repeat/bin_size)
        return Env(input_array,rate=rate,delay=self.delay,tjust=self.tjust)   

    def _envelope_binmin(self, bin_size=35, repeat=1):
        """
        Calculate envelope as min value in bins of 'bin_size' points, repeating each value 'repeat' times
        Note that the abs is not calculated prior to max. This usually returns negative numbers. 
        """
        input_array=numpy.array(list(map(min,numpy.array_split(self,math.floor(self.samples/bin_size))))).repeat(repeat)
        rate=self.rate*(repeat/bin_size)
        return Env(input_array,rate=rate,delay=self.delay,tjust=self.tjust)
        
    def _envelope_binabs(self, bin_size=35, repeat=1):
        """
        Calculate envelope as max of abs value in bins of 'bin_size' points, repeating each value 'repeat' times
        """
        input_array=numpy.array(list(map(max,numpy.array_split(abs(self),math.floor(self.samples/bin_size))))).repeat(repeat)
        rate=self.rate*(repeat/bin_size)
        return Env(input_array,rate=rate,delay=self.delay,tjust=self.tjust)  
        
    def _envelope_robust(self, percentile=95, bin_size=440, repeat=1, cut_freq=None, padtype=None, **kwargs):
        """
        Calculate the envelope based on the a robust version of the Jarne method
        binmap with percentile of the absoulte value and lowpass Butterworth order 4 filter
        
        kwargs - dict
            further arguments for scipy.signal.filtfilt
        """
        robust_func=lambda x:numpy.percentile(numpy.abs(x),percentile)
        if cut_freq is None:
            return self._envelope_binmap(func=robust_func,bin_size=bin_size,repeat=repeat)
        else:
            return self._envelope_binmap(func=robust_func,bin_size=bin_size,repeat=repeat).butter(high=cut_freq,padtype=padtype,**kwargs)
            
   
    def _envelope_teager(self):
        """
        Teager-Kaiser energy operator
        """
        a=self.array            
        input_array=a[1:-1]*a[1:-1]-a[2:]*a[:-2]
        return Env(input_array,rate=self.rate,delay=self.delay+1/self.rate,tjust=self.tjust)      
        
    def _envelope_robust_teager(self, percentile=95, bin_size=440, repeat=1, cut_freq=None, padtype=None, **kwargs):
        """
        Enenvelope calculated by calculating the Teager-Kaiser energy operator,
        binning, calculating a percentile and (optinally) applying a low pass Butterworth filter
        """
        robust_func=lambda x:numpy.percentile(x,percentile)
        if cut_freq is None:
            return self._envelope_teager()\
                ._envelope_binmap(func=robust_func,bin_size=bin_size,repeat=repeat)
        else:
            return self._envelope_teager()\
                ._envelope_binmap(func=robust_func,bin_size=bin_size,repeat=repeat)\
                .butter(high=cut_freq,padtype=padtype,**kwargs)
                
    def _envelope_log_robust_teager(self, percentile=95, bin_size=220, **kwargs):
        """
        Enenvelope calculated as log10 of 'robust_teager'
        """
        return numpy.log10(self._envelope_robust_teager(percentile=percentile,bin_size=bin_size,**kwargs))                

        
    def _envelope_jarne(self, bin_size=35, repeat=1, cut_freq=100, padtype=None, **kwargs):
        """
        Calculate the envelope based on the Jarne method: binmax and lowpass Butterworth order 4 filter
        
        kwargs - dict
            further arguments for scipy.signal.filtfilt
        """
        return self._envelope_binabs(bin_size=bin_size,repeat=repeat).butter(high=cut_freq,padtype=padtype,**kwargs)
        
    def _envelope_mindlin(self, tau=100, sg_window=513, sg_order=4):
        """
        Calculate de envelope by the Mindlin method: Hilbert transform, LTI integration and Savitzky-Golay filter
        
        tau - integer
            Characteristic time of the decay of the LTI integration, mesured in samples
            
        sg_window - odd integer
            Size of the window used in the Savitzky-Golay filter. 
            
        sg_order - integer
            Order of the polynomial used in the Savitzky-Golay filter. 
        """
        #Calculating Hilbert transform and modulus of the analytic function
        h = numpy.abs(scipy.signal.hilbert(self))
        
        #calculating Linear-Time-Invariant integral
        #discrete version of dx/dt=-(1/tau)*x+h
        # x[n+1] = A*x[n] + B*h[n] #state variable
        # y[n] = C*x[n] + D*h[n]   #output
        # A=(1-1/tau), B=1, C=1/tau, D=0 
        dLTI=scipy.signal.StateSpace((1-1/tau),1,1/tau,0,dt=1) #defining the system
        _, ih, _ = scipy.signal.dlsim(dLTI,h) #running the integration
        
        #Applying Savitzky-Golay filter
        sgih=scipy.signal.savgol_filter(ih.reshape(-1), sg_window, sg_order) 
        
        return Env(sgih,rate=self.rate,delay=self.delay,tjust=self.tjust)
        
    def _envelope_absint(self, tau=44):
        """
        Calculate de envelope as the LTI of the absolute value
        
        tau - integer
            Characteristic time of the decay of the LTI integration, mesured in samples
            
        """
        #calculating Linear-Time-Invariant integral
        #discrete version of dx/dt=-(1/tau)*x+h
        # x[n+1] = A*x[n] + B*h[n] #state variable
        # y[n] = C*x[n] + D*h[n]   #output
        # A=(1-1/tau), B=1, C=1/tau, D=0 
        dLTI=scipy.signal.StateSpace((1-1/tau),1,1/tau,0,dt=1) #defining the system
        _, ih, _ = scipy.signal.dlsim(dLTI,numpy.abs(self.array)) #running the integration
        return Env(ih.reshape(-1),rate=self.rate,delay=self.delay,tjust=self.tjust)
       
    def spectrogram(self, method="scipy", tlim=None, **kwargs):
        """
        Calculate the spectrogram of the Wav with the indicated method
        
        
        returns a Spe object. Note that the log10 of the spectrum returned.
        zeros are replaced by eps to avoid numerical issues prior to the log10 calculation.
        """
        if not method in ['scipy','mlab','scipy_praat','mlab_praat']: raise Error("Method "+method+" not implemented")
        if not tlim is None:
            if len(tlim)!=2:
                raise Error("Wav.spectrogram tlim should be a 2-tuple of floats")
            wav=self.crop(start=tlim[0],end=tlim[1])
        else:
            wav=self
        method_name="_spectrogram_"+method.lower()
        spectogram_method=getattr(wav,method_name)
        (spectrum, freqs, times)=spectogram_method(**kwargs)    
        spectrum[spectrum<=numpy.finfo(numpy.float64).eps]=numpy.finfo(numpy.float64).eps #replacing zero with eps to avoid errors
        self.spe=Spe(numpy.log10(spectrum),freqs,times+wav.delay)
        #self.spe=Spe(spectrum,freqs,times+wav.delay)
        self.spe.method=method
        kwargs.update({'tlim':tlim})
        self.spe.method_args=kwargs
        return self.spe 
        
    def _spectrogram_scipy_praat(self,view_range=(0,11000),window_length=0.01,time_steps=1000,\
                            dynamic_range=70,window=None,scaling='density',**kwargs):
        nperseg=int(self.rate*window_length*2)
        noverlap=int(nperseg-self.samples/time_steps)
        noverlap=max([noverlap,0]) #the minimum overlap has to be zero
        noverlap=min([noverlap,int(nperseg*(1-1/(8*math.sqrt(math.pi))))]) #the max overlap is nperseg/(8*sqrt(pi))
        if window is None: window=('gaussian',nperseg)
        (f, t, Sxx)=scipy.signal.spectrogram(self, fs=self.rate, nperseg=nperseg, noverlap=noverlap, window=window, \
                                                   scaling=scaling, **kwargs)
        Sxx.clip(min=float(Sxx.max()*10**(-abs(dynamic_range)/10)),out=Sxx)
        return (Sxx, f, t)

        
    def _spectrogram_scipy(self,window='barthann',nperseg=512,nfft=None,noverlap=448,scaling='spectrum',**kwargs):
        kwargs.update({'window':window, 'nperseg':nperseg, 'nfft':nfft, 'noverlap':noverlap, 'scaling':scaling})
        (f, t, Sxx)=scipy.signal.spectrogram(self, fs=self.rate, **kwargs)
        return (Sxx, f, t)

    def _spectrogram_mlab_praat(self,view_range=(0,11000),window_length=0.01,dynamic_range=70,**kwargs):    
        noverlap = int(self.rate*window_length)
        NFFT = noverlap*2 
        (spectrum, freqs, t)=matplotlib.mlab.specgram(self, Fs=self.rate, NFFT=NFFT, noverlap=noverlap, **kwargs)
        spectrum.clip(min=float(spectrum.max()*10**(-abs(dynamic_range)/10)),out=spectrum)
        return (spectrum, freqs, t)
         
    def _spectrogram_mlab(self, **kwargs):
        """
        spectrogram(method='mlab',NFFT=512,noverlap=256)
        """
        return matplotlib.mlab.specgram(self, Fs=self.rate, **kwargs)

    def correlate(self, template, method="pearson", **kwargs):
        """
        Calculate the correlation with a template
        
        template - Wav or Env object with which to correlate
        method - string indicating the method o use ['pearson', 'pearson_numba', 'pearson_global', 'dot' or 'similiude']
    
        key word arguments
        mode[numpy] - string indicating mode of correlation. 'valid', 'full' or 'same' (numpy.correlate)
         """
        if not abs(self.rate - template.rate) < self.rate/1000:
            raise Error("Attemping to correlate self.rate="+str(self.rate)+" with template.rate="+str(template.rate))
        method=method.lower()
        method_name="_correlate_" + method
        if not method in ['pearson','pearson_numba','pearson_global','similitude']: raise Error("Method "+method+" not implemented")
        correlate_method=getattr(self,method_name)
        cor=correlate_method(template, **kwargs)
        cor.method=method
        cor.method_args=kwargs
        cor.channel=self.channel + "_correlate"
        return cor
    
    def _correlate_similitude(self, template, similitude='(1-d)/(1+d)', distance='NED', mode='valid'):
        n1=len(template)
        assert n1<=len(self)         
        t=template.view(numpy.ndarray)
        
        d=None
        if distance=='SNED' or distance=='NED':
            u=numpy.ones(n1)      
            d=1-2*numpy.correlate(self.array,t,mode=mode)/                   \
            (numpy.correlate(numpy.square(self.array),u,mode=mode)+numpy.sum(numpy.square(t)))   
            if distance=='NED':
                d=numpy.sqrt(d)
        else:
            raise Error('unknown distance ' + distance + ' for _correlate_similitude')
         
        s=None
        if similitude=='1/(1+d)':
            s = 1/(1+d)
        elif similitude=='(1-d)/(1+d)':
            s = (1-d)/(1+d)
        else:
            raise Error('unknown similitude ' + similitude + ' for _correlate_similitude')
        
        if mode == 'full':
            delay=-template.duration
        else:
            delay=0
            
        return Cor(s,self.rate,delay=delay,tjust=self.tjust)  
        
    def _correlate_pearson(self, template, mode='valid'):
        """
        Calculate the correlation with a template
        
        template - Wav or Env
            object with which to correlate
            
        mode - string indicating mode of correlation. 'valid', 'full' or 'same' (numpy.correlate)
            
        returns a vector Wav of self.samples-len(template) samples with correlation coefficient
            calculates as r = sum(x1*x2)/n1 where x1=(template-mean)/sd and x2=(self-mean)/sd  
        """
        
        n1=len(template)
        assert (n1<=len(self) or mode!='valid')
        std_template=numpy.std(template.array)
        if std_template > numpy.finfo(float).eps:
            x1=(template.array-numpy.mean(template.array))/std_template
            u=numpy.ones(n1)
            r=numpy.correlate(self,x1,mode=mode)/numpy.sqrt(n1*numpy.correlate(numpy.square(self),u,mode=mode)-numpy.square(numpy.correlate(self,u,mode=mode)))
            #substracting two large, similar numers and then calculating sqrt. 
            if mode == 'full':
                delay=-template.duration
            else:
                delay=0
            return Cor(r,self.rate,delay=delay)    
        else:
            return Cor.zeros(samples=self.samples - len(template) + 1,rate=self.rate) 

    def _correlate_pearson_global(self, template, mode='valid'):
        """
        Calculate the correlation with a template, assuming pre-normalization
        
        template - Wav or Env
            object with which to correlate
            
        mode - string indicating mode of correlation. 'valid', 'full' or 'same' (numpy.correlate)
            
        returns a vector Wav of self.samples-len(template) samples with correlation coefficient
            calculates as r = sum(x1*x2)/n1 where x2=template and x1=self (proper normaliztion depends on you!)
        """
        n2=len(template.array)
        assert (n2<=len(self) or mode!='valid')
        r=numpy.correlate(self.array,template.array,mode=mode)/n2
        if mode == 'full':
            delay=-template.duration
        else:
            delay=0
        return Cor(r,self.rate,delay=delay)    
            
    def _correlate_pearson_numba(self, template):
        template = template.view(numpy.ndarray)
        std_template=numpy.std(template)
        if std_template > numpy.finfo(float).eps:
            norm_x1 = (template - numpy.mean(template))/std_template
            return Cor(_correlate1D(norm_x1,self.array),self.rate,tjust=self.tjust)
        else:
            return Cor.zeros(samples=self.samples - len(template) + 1,rate=self.rate,tjust=self.tjust) 

    def cor(self,to):
        """
        calculates pearson correlation coefficient for aligned segment of self and to 
        
        to - Wav
            Wav object to which correlate
        """
        if not abs(self.rate - to.rate) < numpy.finfo(float).eps*100:
            to=to.resample(self.rate)
        start=max(self.start,to.start)
        end=min(self.end,to.end)
        samples=int((end-start)*self.rate)
        x1=self.crop(start,samples=samples)
        x2=to.crop(start,samples=samples)
        mask=numpy.logical_not(numpy.logical_or(numpy.isnan(x1.array),numpy.isnan(x2.array))) 
        x1=x1[mask].normalize(method='meanstd').array
        x2=x2[mask].normalize(method='meanstd').array
        return numpy.dot(x1,x2)/len(x1)
        
            
    def correlation_spectrogram(self,to,window=0.1,step=0.01,tau_max=None,metric='similitude',type_='parallel',normalize=None,**kwargs):
        """
        Calculates a Correlation Spectrogram 
        
        calculates the correlation between self and template for different times and taus (delays)
        
        to - Wav
            the template to which cross correlate
        window - float
            the window size in soconds with which to do the correlation
        step - float
            time step in seconds or the analysis 
            for parallel cross-correlation defines output rate
            for orthogonal cros-correlation defines de y axis
        tau_max - float
            maximal time delay (tau, positive and negative) for the cross-correlation
            used for parallel cross-correlation  only
        metric - string
            defines the method of the correlation ('pearson', 'pearon_numba', 'pearson_global' or 'similitude')
            see Wav.correlate
        type_ - string
            defines the type of correlation spectrogram
            'parallel' gives the delays (taus) vs time
            'orthogonal' gives a matrix of correlation vectors for each window of the template
        kwargs 
            further arguments for Wav.correlate
            
        Returns an object of type CorSep (inherits from TimeMatrix)
        """
        assert isinstance(to, Wav)        
        template=to
        
        if type_=='parallel':
            if tau_max is None: tau_max = window
            
            #start = self.start + window/2 + tau_max + 1/self.rate
            #end = self.end - window/2 - tau_max - 1/self.rate
            start = self.start + window/2
            end = self.end - window/2
            
            X1=to
            X2=self
            if metric == 'pearson_global' and normalize is None:
                normalize = 'meanstd'
            if normalize is not None:
                X1 = X1.normalize(method=normalize)
                X2 = X2.normalize(method=normalize)
            #template=to.zero_pad(min(to.start,self.start - 2*tau_max - window/2),max(to.end,self.end + 2*tau_max + window/2))
            #X1=X1.zero_pad(min(X1.start,self.start - tau_max - window/2 - 1/X1.rate),max(X1.end,self.end + tau_max + window/2 + 1/X1.rate))
            
            window_samples=int(window*X1.rate)
            tau_max_samples=int(tau_max*X1.rate)
            X1_samples=window_samples+2*tau_max_samples#-1
            l_cor=list()
            times = numpy.arange(start,end,step)
            cor=None
            taus=None
            for j_start in times:
                #j_start = start
                #j_start = times[-1]
                X1_left = j_start-tau_max-window/2#-1/X1.rate                
                X1_right= X1_left + X1_samples/X1.rate
                if X1_left < X1.start:
                    X1_j=X1.crop(X1.start,samples=X1_samples-int((X1.start-X1_left)*X1.rate))
                    cor=X1_j.correlate(X2.crop(j_start-window/2,samples=window_samples),method=metric,**kwargs).array
                    cor=numpy.append(numpy.repeat(numpy.nan, X1_samples-X1_j.samples),cor)
                elif X1_right > X1.end:
                    X1_j=X1.crop(X1_left,samples=X1_samples-int((X1_right-X1.end)*X1.rate))
                    cor=X1_j.correlate(X2.crop(j_start-window/2,samples=window_samples),method=metric,**kwargs).array
                    cor=numpy.append(cor,numpy.repeat(numpy.nan, X1_samples-X1_j.samples))
                else:
                    X1_j=X1.crop(X1_left,samples=X1_samples)
                    cor=X1_j.correlate(X2.crop(j_start-window/2,samples=window_samples),method=metric,**kwargs).array

                l_cor.append(cor)
            #[len(i) for i in l_cor]
            taus = numpy.arange(-tau_max_samples,tau_max_samples+1)/X1.rate   
            cor_spe=CorSpe(numpy.transpose(numpy.fliplr(numpy.array(l_cor))),taus,times)

        elif type_=='orthogonal':
            window_samples=int(window*template.rate)
            l_cor=list()
            template_times = numpy.arange(template.start+window/2,template.end-window/2,step)
            cor=None
            z=Cor.zeros(rate=1/step,duration=self.duration,delay=self.delay+window/2)
            for i_start in template_times:
                #i_start = template_times[0]
                #i_start = template_times[-1]
                i_template=template.crop(i_start-window/2,samples=window_samples)
                cor=z+self.correlate(i_template,method=metric,**kwargs).set_delay(self.delay+window/2)
                l_cor.append(cor.array)
            #[len(i) for i in l_cor]
            times = cor.get_times()    
            #cor_spe=CorSpe(numpy.fliplr(numpy.flipud(numpy.array(l_cor))),template_times,times)
            cor_spe=CorSpe(numpy.array(l_cor),template_times,times) #no entiendo porque esto funciona

        else:
            raise Error("unknow type_. Should be 'parallel' or 'orthogonal'.")

        cor_spe.method='correlation_spectrogram'
        cor_spe.method_args=dict({'template':template,'window':window, \
            'step':step,'tau_max':tau_max,'start':start,\
            'end':end,'metric':metric,'type_':type_})
                        
        return cor_spe

        
    def time_warp(self,*args,**kwargs):
        return self.stw(*args,**kwargs)
        
    def stw(self,to=None,delays=None,dinterval=0.100,dmax=0.050,dstep=0.010,**kwargs):
        """
        Applies a smooth time warping to the vector in order to maximize the correlation to 'to'
        
        to - Wav, Env or None
            pattern to which to maximize the correlation by time warping
            if None 'delays' should be given
        delays - list of floats, numpy array, string, callable or None
            list of delays to apply in the time warping. len(delays) should be self.duration//dinterval+2
            if 'to' is None, the provided delays are applied as is, if not they are used as starting points in 
            the time warping fitting algorithm.
            if string, should be transformed to float array by numpy.fromstring(delays[1:-1],sep=' ')
            if None, 'to' should be given.
        dinterval - float
            time interval in seconds between consecutive delay parameters
        dmax - float
            maximum delay 'allowed'
        dstep - float
            penalization factor for appartment from lineality 
        kwargs - dict
            further arguments to scipy.optimize.minimize
            
        Returns a Warped (Wav) object with the following attributes:
            stw_times - np.array
            stw_delays - np.array
            stw_times_delays - np.array (dim=2)
            stw_delays_str - str
            stw_times_delays_str - str
            stw_fun - callable
            stw_delay_fun - callable
            stw_cor - float
        """
        
        assert to is not None or delays is not None
        
        v=self.normalize(method='meanstd')
        
        f=v.interpolate(kind='cubic')
        M=v.duration//dinterval+2
        td=v.start+v.duration/2+dinterval*(numpy.arange(M)-(M-1)/2)
        
        if delays is not None:
            if isinstance(delays,str):
                delays_str=delays
                try:
                    delays=delays_str[1:-1]
                    if '\n' in delays:
                        dsl=delays.split('\n ')
                        delays=numpy.array([numpy.fromstring(dsl[0][1:-1],sep=' '),\
                                            numpy.fromstring(dsl[1][1:-1],sep=' ')])
                    else:
                        delays=numpy.fromstring(delays,sep=' ')
                except:
                    raise Error("could not parse delays='"+delays_str+"'")
            if isinstance(delays,list):
                delays = numpy.array(delays)
            if isinstance(delays,numpy.ndarray):
                if delays.ndim==1:
                    if len(delays)!=M:
                        raise Error("len(delays) should be equal to M=self.duration//dinterval+2")
                elif delays.ndim==2:
                    delays=scipy.interpolate.interp1d(delays[0,:],delays[1,:],\
                                               kind='cubic',bounds_error=False,\
                                               fill_value=(delays[1,0],delays[1,-1]))
                else:
                    raise Error("delays should be a np.ndarray of dimension 1 or 2")
            if callable(delays):
                delays = delays(td)
        
        if to is not None:  
            p=(type(self).zeros(template=self)+to).normalize(method='meanstd')
            
            if delays is None:
                if v.duration < 2*dmax:
                    raise Error("self duration should be at least twice dmax")
                pvcor=p.correlate(v.crop(v.start+dmax,v.end-dmax)).set_delay(-dmax)
                d=numpy.repeat(-pvcor.time_max(),M)
            else:
                d=numpy.array(delays)
                
            k0 = (1/dmax)**4
            k1 = (1/dstep)**2
        
            def costfun(d):
                w=scipy.interpolate.interp1d(td,td+d,kind='cubic')
                s=Env(f(w(v.get_times())),v.rate,v.delay)
                cost = -numpy.dot(s,p)/len(s)
                cost+= k0*numpy.sum(numpy.power(d,4))/len(d)
                cost+= k1*numpy.sum(numpy.square(d[1:-1]-(d[0:-2]+d[2:])/2))/len(d)
                return cost
            
            d1=scipy.optimize.minimize(costfun,d,**kwargs).x
        else:
            d1=delays

        w1=scipy.interpolate.interp1d(td,td+d1,kind='cubic')
        f1=self.interpolate(kind='cubic',fill_value=numpy.nan)
        s1=Warped(f1(w1(self.get_times())),self.rate,self.delay)
   
        s1.method='time_warp'
        s1.method_args=kwargs
        s1.method_args['to']=to
        s1.method_args['dinterval']=dinterval
        s1.method_args['dmax']=dmax
        s1.method_args['dstep']=dstep
        s1.stw_times=td
        s1.stw_delays=d1
        s1.stw_cor=numpy.nan
        if to is not None:
            s1.stw_cor=s1.cor(to)
        
        #these guys could be calculated from stw_times and stw_delays (implemented as Warped methods)            
        #s1.stw_times_delays=numpy.array([td,d1])
        #s1.stw_delays_str=numpy.array2string(d1,max_line_width=1e6,precision=4)
        #s1.stw_times_delays_str=numpy.array2string(s1.stw_times_delays,max_line_width=1e6,precision=4)
        #s1.stw_fun=w1
        #s1.stw_delay_fun=scipy.interpolate.interp1d(td,d1,kind='cubic')
        return s1                


    def plot(self,tlim=None,*args,**kwargs):
        """
        Plot samples vs time using matplotlib.pyplot.plot
        
        tlim - None, float or 2-tuple
            sets the time range of the plot
        """
        matplotlib.pyplot.plot(self.get_time(),self,*args,**kwargs)
        if tlim is not None: 
            if isinstance(tlim,tuple):
                matplotlib.pyplot.xlim(tlim)
            else:
                matplotlib.pyplot.xlim(self.start-tlim,self.end+tlim)
        if self.channel != "": matplotlib.pyplot.ylabel(self.channel)
        matplotlib.pyplot.xlabel("time (s)")
        return self
        
    def plot_welch(self, nperseg=512, *args, **kwargs):
        """
        Plot the Welch power spectrum of the Wav
        """
        f1, Pxx_den = scipy.signal.welch(self, self.rate, nperseg=nperseg)
        matplotlib.pyplot.plot(f1, numpy.log10(Pxx_den), *args, **kwargs)
        return self
        
    def lfilter(self, b, a=1, *args, **kwargs):
        """
        Filter Wav with an IIR or FIR filter.
        
        b : array_like
            The numerator coefficient vector in a 1-D sequence.
        a : array_like
            The denominator coefficient vector in a 1-D sequence. If a[0] is not 1, then both a and b are normalized by a[0].    
        """
        return Wav(scipy.signal.lfilter(b, a, self, *args, **kwargs),self.rate,self.delay,self.channel)
        
    def butter(self,low=None,high=None,order=4,btype=None,padtype=None,causal=False,**kwargs):
        """
        Applies a forward-backward non-causual Butterworht digital filter. 
        low - float
            lower frequency cut in Hz
        high - float
            higher frequency cut in Hz
        order - int
            order of the Butterworht filter
        btype - None, 'lowpass', 'highpass', 'bandpass' or 'bandstop'
            If None the btype is infered from low and high. If high is given a lowpass filter is applied.
            If low is given a highpass filter is applied. If both are given and low<high a bandpass filter is applied. 
            If high<low a band stop filter is applied.
        padtype - str or None, optional
            Must be ‘odd’, ‘even’, ‘constant’, or None. 
            This determines the type of extension to use for the padded signal to which the filter is applied. 
            If padtype is None, no padding is used. The default is ‘odd’.   
            Used for non-causal filter only
        causal - boolean
            If False a forward-backward non-causual filter is applied.
            If True a causal filter is applied. 
        Returns a filtered Wav object 
        """
        #assertions
        if low is None and high is None: raise Error("low and high cannot be None")
        #defining filter type
        if btype is None:
            if low is None: btype='lowpass'
            elif high is None: btype='highpass'
            else:
                if low < high: btype='bandpass'
                else: btype='bandstop'; low, high = high, low
        #defining Wn parameter
        nyq = float(self.rate*0.5) #Nyquist frequency
        if btype == 'lowpass': Wn=float(high/nyq)    
        elif btype == 'highpass': Wn=float(low/nyq) 
        else: Wn=[float(low/nyq), float(high/nyq)]
        #running filter
        b, a = scipy.signal.butter(order, Wn, btype=btype) 
        if causal is False:
            return type(self)(scipy.signal.filtfilt(b,a,self,padtype=padtype,**kwargs),self.rate,self.delay)
        else:
            return type(self)(scipy.signal.lfilter(b,a,self,**kwargs),self.rate,self.delay)
        
    def decimate(self, q=10, zero_phase=True, **kwargs):
        """
        Downsample the signal after applying an anti-aliasing filter (see scipy.signal.decimate).
        q - int, optional
            The downsampling factor. Defaults to 10. For downsampling factors higher than 13, it is recommended to call decimate multiple times.
        zero_phase : bool, optional
            Prevent phase shift by filtering with filtfilt instead of lfilter when using an IIR filter, and shifting the outputs back by the filter’s group delay when using an FIR filter. 
        kwargs - dict
            Further arguments for scipy.signal.decimate
            
        Returns a decimated Wav object    
        """
        return Wav(scipy.signal.decimate(self, q=q, zero_phase=zero_phase, **kwargs), self.rate/q, self.delay, tjust=self.tjust)

    def play(self,start=None,end=None):
        """
        Play the wav signal as sound
        start - double, optinal
            start time in seconds
        end - double, optional
            end time in seconds
            
        uses mplayer
        Installation:
            ubuntu> sudo apt-get install mplayer
        """
        if start is None: start=0
        if end is None: end=self.duration
        fn=os.path.join(self.file_path,self.file_name)
        if not os.path.isfile(fn):
            f=tempfile.NamedTemporaryFile(mode='w', suffix='.wav', prefix='tmp', delete=False)
            self.write(f.name,log=None)
            fn=f.name
        cmd="mplayer -ss " + str(start) + " -endpos " + str(end-start) + " " + fn + " &"
        os.system(cmd)
        return self

    def zero_pad(self,start=0,end=None):
        """
        zero pad the Wav
        
        start - float
            time point at which the zero-padded Wav object should start
        end - float
            time point at which the zero-padded Wav object should end
            If None is set to the end of the wav (no trailing zeros)            
        """
        if end is None:
            end = self.end
        z=type(self).zeros(rate=self.rate,duration=end-start,delay=start,tjust=self.tjust)
        return z.__add__(self.crop(start,end))
        
    def nan_pad(self,start=0,end=None):
        """
        nan pad the Wav
        
        start - float
            time point at which the zero-padded Wav object should start
        end - float
            time point at which the zero-padded Wav object should end
            If None is set to the end of the wav (no trailing zeros)            
        """
        if end is None:
            end = self.end
        z=type(self).zeros(rate=self.rate,duration=end-start,delay=start,tjust=self.tjust)
        z.fill(numpy.nan)
        return z.__add__(self.crop(start,end))
        
        
    def interpolate(self,times=None,bounds_error=False,fill_value=0,**kwargs):
        """
        Returns a linear interpolation of the Wav
        
        times - numpy.array or None
            if times is None returns the interpolation function
            if it is a numpy.array it evaluates the interpolation function in this array and returns the results
        bounds_error - bool
            see documentation for scipy.interpolate.interp1d
        fill_value - float
            see documentation for scipy.interpolate.interp1d
        kwargs - dict
            further arguments for scipy.interpolate.interp1d
            
        if times is None, returns a scipy.interpolate.interp1d interpolation function
        if times is a numeric vector, reuturns a numeric vector
        """
        x=self.get_time()
        y=self.view(numpy.ndarray)
        if abs(self.tjust)<numpy.finfo(float).eps: #tjust==0
            x=numpy.append(x,x[-1]+1/self.rate)                    
            y=numpy.append(y,y[-1])                    
        elif abs(1-self.tjust)<numpy.finfo(float).eps: #tjust==1
            x=numpy.insert(x,0,x[0]-1/self.rate)                    
            y=numpy.insert(y,0,y[0])                    
        else:
            x=numpy.append(x,x[-1]+(1-self.tjust)/self.rate)                    
            y=numpy.append(y,y[-1])                    
            x=numpy.insert(x,0,x[0]-self.tjust/self.rate)                    
            y=numpy.insert(y,0,y[0])                    
        if times is None:
            return scipy.interpolate.interp1d(x, y,bounds_error=bounds_error,fill_value=fill_value,**kwargs)
        else:
            return scipy.interpolate.interp1d(x, y,bounds_error=bounds_error,fill_value=fill_value,**kwargs)(times)
            
    def resample(self,rate):
        """
        Resamples Wav to target rate
        """        
        if abs(self.rate - rate) < 10*numpy.finfo(float).eps:
            return self
        else:
            return type(self).zeros(rate=rate,duration=self.duration,delay=self.delay,tjust=self.tjust).__add__(self)
            
    def __add__(self,other):
        if isinstance(other,Wav):
            #using linear interpolation
            a,b = self.array,other.interpolate(self.get_times(),fill_value=numpy.nan)
            na,nb = numpy.isnan(a), numpy.isnan(b)
            s=numpy.nansum(numpy.stack((a,b)),0)
            na &= nb
            s[na] = numpy.nan
            return type(self)(s,rate=self.rate,delay=self.delay, tjust=self.tjust)
        else:
            return super().__add__(other)

    def __sub__(self,other):
        if isinstance(other,Wav):
            #using linear interpolation
            s1=numpy.nansum(numpy.stack((self.array,-other.interpolate(self.get_times()))),0)
            return type(self)(s1,rate=self.rate,delay=self.delay, tjust=self.tjust)
        else:
            return super().__sub__(other)
            
        
class Env(Wav): 
    """
    Object represeting a envelope of a WAV file. 
    
    Inherits from Wav. Additional properties:
    method: string
        Name of the method used to calculate the envelope
    method_args: dict
        Arguments used when calculating the envelope
    """
    pass        
        
class Fund(Wav): 
    """
    Object represeting the fundamental requency of a WAV file. 
    
    Inherits from Wav. Additional properties:
    method: string
        Name of the method used to calculate the envelope
    method_args: dict
        Arguments used when calculating the envelope
    """
    pass        
    
class Cor(Wav): 
    """
    Object represeting a correlation between a Wav and a template
    
    Inherits from Wav. 
    """
    pass        
            
class Warped(Wav): 
    """
    Object represeting a time warped Wav or Env 
    
    Inherits from Wav. Has additional attributes
        method - str
        method_args - dict
        stw_delay_times - numpy array fo floats
        stw_delays - numpy array fo floats
        stw_delays_str - str
        stw_fun - callable
        stw_delay_fun - callable
        stw_cor - float 
    """
    @property
    def stw_times_delays(self):
        return numpy.array([self.stw_times,self.stw_delays])
        
    @property
    def stw_delays_str(self):
        return numpy.array2string(self.stw_delays,max_line_width=1e6,precision=4)
    
    @property
    def stw_times_delays_str(self):
        return numpy.array2string(self.stw_times_delays,max_line_width=1e6,precision=4)
    
    @property
    def stw_fun(self):
        return scipy.interpolate.interp1d(self.stw_times,self.stw_times+self.stw_delays,kind='cubic')
    
    @property
    def stw_delay_fun(self):
        return scipy.interpolate.interp1d(self.stw_times,self.stw_delays,kind='cubic')
        
    @property
    def stw_min_delay(self):
        return numpy.amin(self.stw_delays)        

    @property
    def stw_max_delay(self):
        return numpy.amax(self.stw_delays)        
        
    @property
    def annot_values(self):
        return {'stw_cor':self.stw_cor,'stw_delays_str':self.stw_delays_str,\
                'stw_times_delays_str':self.stw_times_delays_str,\
                'stw_min_delay':self.stw_min_delay,'stw_max_delay':self.stw_max_delay}
            
    def get_delays_env(self):
        return Env(self.stw_delay_fun(self.get_times()),rate=self.rate,\
                   delay=self.delay,tjust=self.tjust,channel="stw_delay")

    def __add__(self,other):
        out = super().__add__(other)
        out.method = self.method
        out.method_args = self.method_args
        if hasattr(self, 'stw_times'):
            out.stw_times = self.stw_times
        if hasattr(self, 'stw_delays'):
            out.stw_delays = self.stw_delays
        if hasattr(self, 'stw_cor'):
            out.stw_cor = self.stw_cor
        return out

    def __sub__(self,other):
        out = super().__sub__(other)
        out.method = self.method
        out.method_args = self.method_args
        if hasattr(self, 'stw_times'):
            out.stw_times = self.stw_times
        if hasattr(self, 'stw_delays'):
            out.stw_delays = self.stw_delays
        if hasattr(self, 'stw_cor'):
            out.stw_cor = self.stw_cor
        return out

    def __mul__(self,other):
        out = super().__mul__(other)
        out.method = self.method
        out.method_args = self.method_args
        if hasattr(self, 'stw_times'):
            out.stw_times = self.stw_times
        if hasattr(self, 'stw_delays'):
            out.stw_delays = self.stw_delays
        if hasattr(self, 'stw_cor'):
            out.stw_cor = self.stw_cor
        return out

    def __div__(self,other):
        out = super().__div__(other)
        out.method = self.method
        out.method_args = self.method_args
        if hasattr(self, 'stw_times'):
            out.stw_times = self.stw_times
        if hasattr(self, 'stw_delays'):
            out.stw_delays = self.stw_delays
        if hasattr(self, 'stw_cor'):
            out.stw_cor = self.stw_cor
        return out
    
class Promise(collections.abc.Callable):
    pass

class WavPromise(Promise):
    
    def __init__(self,path,name,channel="",delay=0,tjust=0.5,norm_max=1.0,norm_min=-1.0,norm_factor=0.999,norm=True,preprocess=None):
        self.file_path=str(path)
        self.file_name=str(name)
        self.channel=str(channel)
        self.delay=float(delay)
        self.tjust=float(tjust)
        self.norm_max=float(norm_max)
        self.norm_min=float(norm_min)
        self.norm_factor=float(norm_factor)
        self.norm=bool(norm)
        self.preprocess=preprocess
        self.crop_start=None
        self.crop_end=None
        self.crop_samples=None
        self.crop_align=None
        self.cropped=False
        self.zero_pad_start=None
        self.zero_pad_end=None
        self.zero_padded=False
        
    def __contains__(self, item):
        return item == self.file_name or item == os.path.join(self.file_path,self.file_name)
        
    def __hash__(self):
        return hash((self.file_path, self.file_name, self.channel,self.delay,self.tjust))
        
    def __len__(self):
        #ToDo: implement this in a more efficient way. See for example
        #http://stackoverflow.com/questions/7833807/get-wav-file-length-or-duration/7842081
        fs, input_array = scipy.io.wavfile.read(os.path.join(self.file_path,self.file_name))
        return input_array.shape[0]       

    def __call__(self):
        fulfil=Wav.read(os.path.join(self.file_path,self.file_name),\
                delay=self.delay, tjust=self.tjust, norm=self.norm, channel=self.channel,\
                norm_max=self.norm_max, norm_min=self.norm_min, norm_factor=self.norm_factor)
        if self.preprocess is not None:
            fulfil=self.preprocess(fulfil)
        if self.cropped:
            fulfil=fulfil.crop(start=self.crop_start,end=self.crop_end,samples=self.crop_samples,aligned=self.crop_aligned)
        if self.zero_padded:
            fulfil=fulfil.zero_pad(start=self.zero_pad_start,end=self.zero_pad_end)
        return fulfil
        
    def __repr__(self):
        s = ("%s(channel=%s, delay=%0.3fsec, tjust=%0.1f)" %\
                (self.__class__.__name__, self.channel, self.delay, self.tjust))
        if len(self.file_name)>0: s=s+"\n"+self.file_name
        if self.preprocess is not None:
            s=s+"\n"+" preprocess="+repr(self.preprocess)
        if self.cropped:
            s=s+"\n"+".crop(start=%s,end=%s,samples=%s,aligned=%s)"%\
                (str(self.crop_start),str(self.crop_end),str(self.crop_samples),str(self.crop_aligned))
        if self.zero_padded:
            s=s+"\n"+".zero_pad(start=%s,end=%s)"%\
                (str(self.zero_pad_start),str(self.zero_pad_end))
        return s

    def rate(self):
        #ToDo: implement this in a more efficient way. See for example
        #http://stackoverflow.com/questions/7833807/get-wav-file-length-or-duration/7842081
        fs, input_array = scipy.io.wavfile.read(os.path.join(self.file_path,self.file_name))
        return fs
        
    def set_delay(self,delay):
        """
        sets the delay of the WavPromise
        """
        if self.cropped:
            raise Error("Setting delay to a cropped WavPromise is not supported")
        self.delay=delay
        return self
        
    def set_start(self,start):
        """
        sets the start time of the WavPromise
        """
        return self.set_delay(start)
        
    def crop(self,start=None,end=None,samples=None,aligned=True):
        if self.zero_padded:
            raise Error("crop should be called before zero_pad for Promisses")
        self.crop_start=start
        self.crop_end=end
        self.crop_samples=samples
        self.crop_aligned=aligned
        self.cropped=True
        return self
        
    def zero_pad(self,start=0,end=None):
        self.zero_pad_start=start
        self.zero_pad_end=end
        self.zero_padded=True
        return self
        
        
@numba.jit('f8[:](f8[:],f8[:])',nopython=True)
def _correlate1D(x1n,x2):
    """
    Private function to implement 1D correlation using numba    
    """
    n1 = len(x1n)
    n2 = len(x2)
    r = numpy.empty(n2-n1+1)

    for j in range(n2-n1+1):
        mean_x2j = 0
        for i in range(n1):
            mean_x2j += x2[i+j]
        mean_x2j /= n1

        std_x2j = 0
        for i in range(n1):
            std_x2j += (x2[i+j]-mean_x2j)**2
        std_x2j = numpy.sqrt(std_x2j/n1)
        
        ncp = 0
        for i in range(n1):
            ncp += x1n[i] * (x2[i+j]-mean_x2j)/std_x2j
        r[j] = ncp/n1       

    return r
    
from Spe import Spe, CorSpe        
        
        
        
        
        
        
        
        
        
        
        
