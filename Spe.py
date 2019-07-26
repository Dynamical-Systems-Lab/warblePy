"""
Spe.py module of WarblePy. This module defines the spectrogram object and methods.

TimeMatrix: Represents a matrix in which each row represents a different time.
Ens: Object representing an ensemble of Wavs or Envs. Inherits from TimeMatrix. 
Spe: Object representing a spectrogram of a Wav. Inherits from TimeMatrix. 
CorSpe: Object representing a correlation spectrogram. Inherits from TimeMatrix. 

---
Author: Alan Bush 
"""
import os
import scipy.io.wavfile 
import scipy.signal
import scipy.interpolate
import numpy
import math
import matplotlib.pyplot 
import matplotlib.colors
import matplotlib.mlab
import pandas
import warnings

class Error(Exception): pass

class TimeMatrix(numpy.ndarray):
    """
    Class represeting a 'time' matrix 
    
    'Spe', 'CorSpe' and 'Ens' inherit from this class
    """
    def __new__(cls, input_array, y, times, tjust=0.5):
        if not input_array.ndim == 2: Error("input_array must have ndim == 2")
        obj = numpy.asarray(input_array).view(cls)
        obj.y = y
        obj.times = times
        obj.tjust=tjust
        obj.method = None
        obj.method_args = None
        obj.db = None
        return obj
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self.y = getattr(obj, 'y', None)
        self.times = getattr(obj, 'times', 0)
        self.tjust = getattr(obj, 'tjust', None)
        self.method = getattr(obj, 'method', None)
        self.method_args = getattr(obj, 'method_args', None)
        self.db = getattr(obj, 'db', None)

    def __repr__(self):
        s = ("%s(rate=%dHz, duration=%0.3fsec, samples=%i, element=%s, delay=%0.3fsec, tjust=%0.1f)" %
                 (self.__class__.__name__, self.rate, self.duration, self.samples, self.shape[0], self.delay, self.tjust))
        if self.db is not None:
            if hasattr(self.db, 'name'):
                db_name=self.db.name
            else:
                db_name='db'
            s+="\n\t"+db_name+"[N="+str(len(self.db))+"]: "    
            s+=numpy.array2string(self.db.columns,max_line_width=100,separator=",",\
                                  formatter={'str_kind':lambda x: " %s" % x})[1:-1].\
                                  replace("\n","\n\t\t")
            s+="\n"      
        return s
        
    def copy(self,order='K'):
        output=type(self)(self.array.copy(order),y=self.y.copy(),times=self.times.copy(),tjust=self.tjust)
        output.method=self.method
        if self.db is not None:
            output.db=self.db.copy()
        if self.method_args is not None:
            output.method_args=self.method_args.copy()  
        return output
        
    def write(self, filename, path=None):
        """
        Writes the TimeMatrix object to disk
        
        filename - str
            base name of the file (no extension)
        
        path - str
            where to write the file. If None the current directory is used.
        """
        
        if path is not None:
            filename = os.path.join(path,filename)
        
        numpy.savez_compressed(filename,array=self.array,y=self.y,times=self.times,tjust=self.tjust)
        
        if self.db is not None:
            self.db.to_csv(filename+"_db.dat",sep='\t',index=True,na_rep='NA')
        
    @classmethod   
    def read(cls, filename, path=None):
        """
        Reads a TimeMatrix object from disk
        
        filename - str
            base name of the file (no extension)
        
        path - str
            where to read the file from. If None the current directory is used.
        """
        if path is not None:
            filename = os.path.join(path,filename)
        fn_npz = filename + ".npz"
        if not os.path.isfile(fn_npz):
            raise Error("File "+ fn_npz + " not found.")
        npz=numpy.load(fn_npz)
        output=cls(npz['array'],y=npz['y'],times=npz['times'],tjust=npz['tjust'])
        
        fn_dat = filename + "_db.dat"
        if os.path.isfile(fn_dat):
            output.db=pandas.read_table(fn_dat)
            if "record_id" in output.db.columns:
                if "annot_id" in output.db.columns:
                    output.db.set_index(['record_id','annot_id'],inplace=True,drop=False)
                else:
                    output.db.set_index(['record_id'],inplace=True,drop=False)

        output.method="read"
        output.method_args={'filename':filename,'path':path}
            
        return output
        
    @property
    def delay(self):
        """
        The delay time, i.e. the start time
        """
        return self.times[0]-self.tjust/self.rate
 
    def shift(self,delay=None,start=None,end=None,by=None):
        """
        returns a time shifted copy of the TimeMatrix
        
        delay - float
            new delay
        start - float
            new start
        end - float
            new end
        by - float 
            time in seconds by which to shif the TimeMatrix relative to current time
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
            return s.set_delay(delay)
        if end is not None:
            return s.set_delay(end - s.duration)
        if by is not None:
            return s.set_delay(s.delay + by)
       
    def set_delay(self, delay):
        """
        sets a new delay for the TimeMatrix
        """
        self.times+=delay+self.tjust/self.rate-self.times[0]
        return self
        
    @property
    def start(self):
        """
        The start of the time represented by the TimeMatrix object.
        
        Note that this might be different from the first element of the 'times' attribute due to the time justification 'tjust'. 
        """
        return self.delay

    @property
    def end(self):
        """
        The end of the time represented by the TimeMatrix object.
        
        Note that this might be different from the last element of the 'times' attribute due to the time justification 'tjust'. 
        """
        return self.times[-1]+(1-self.tjust)/self.rate

    @property
    def tlim(self):
        return (self.start,self.end)
  
    def get_times(self):
        """
        returns vector of times for each data point
        """
        return self.times
      
    @property
    def array(self):
        """
        The time matrix object as a numpy.ndarray
        """
        return self.view(numpy.ndarray)
        
    @property    
    def rate(self):
        """
        Sampling rate in Hz
        """
        return 1/numpy.mean(self.times[1:]-self.times[0:-1])
        
    @property    
    def duration(self):
        """
        Calculated as self.times[-1]-self.times[0]+1/self.rate, i.e. the time 
        'represented' by the TimeMatrix object
        """
        return self.times[-1]-self.times[0]+1/self.rate

    @property
    def samples(self):
        """
        Number of time samples
        """
        return len(self.times)

    def clip_(self,min_,max_):
        """
        Clips the time matrix object
        """
        return type(self)(numpy.clip(self.array, min_, max_), self.y, self.times)

        
    def crop(self,start=None,end=None,samples=None):
        """
        Returns a cropped copy of the object
        
        start: float
            start time in seconds
        end: float
            end time in seconds
        samples: integer
            number of samples to return
        """
        if start is not None: start=float(start)
        if end is not None: end=float(end)
        crop_idx = self._crop_idx(start,end,samples)
        if crop_idx is None: return None
        (start_idx, end_idx) = crop_idx
        output = type(self)(self.array[:,start_idx:end_idx],y=self.y,times=self.times[start_idx:end_idx],tjust=self.tjust)
        output.db = self.db
        return output

        
    def _crop_idx(self,start=None,end=None,samples=None):    
        """
        Internal function to calculate crop indexes
        If interval required is not contained in TimeMatrix, returns None
        """
        assert (start is not None) + (end is not None) + (samples is not None) == 2
        if start is None:
            if samples is None:
                start=0
                start_idx=0
        else:
            start_idx=math.floor((start-self.delay)*self.rate)
            if start_idx <= 0: start_idx = 0
            if start_idx >= self.shape[1]: return None

        if end is None:
            if samples is None:
                end=self.duration
                end_idx=self.shape[1]
            else:
                end_idx = start_idx + samples 
                if end_idx < 0: end_idx = 0
                if end_idx >= self.shape[1]: end_idx = self.shape[1]
        else:
            end_idx=math.floor((end-self.delay)*self.rate)
            if end_idx <= 0: return None
            if end_idx >= self.shape[1]: end_idx = self.shape[1]

        if start is None and samples is not None:
            start_idx = end_idx - samples
            if start_idx < 0: start_idx = 0
            if start_idx >= self.shape[1]: return None
     
        assert start_idx < end_idx    
        return (start_idx, end_idx)        
        
        
    @classmethod     
    def zeros(cls,template=None,times=None,y=None,rate=None,duration=None,samples=None,delay=0,tjust=0.5):
        """
        Creates a TimeMatrix object filled with zeros

        template - TimeMatrix 
            object from which to copy the dimensions
        times - numpy array of dimension one 
            defines the times of the object. Overwrites times from template 
        y - numpy array of dimension one 
            defines the 'y' dimension (freqs, taus). Overwrites y from template. 
        rate - float
            sampling rate to define time vector. Used if times and template are None.
            if used, either samples or duration should be given as well
        duration - float
            duration in seconds.  Used if times and template are None.
            if used, either samples or rate should be given as well.
            Calculated as self.times[-1]-self.times[0]+1/self.rate, i.e. the time 
            'represented' by the TimeMatrix object
        samples - integer
            number of time samples. Used if times and template are None.
            if used, either duration or rate should be given as well
        delay - float
            time delay in seconds
        """
        assert template is not None or times is not None or ((rate is not None) + (duration is not None) + (samples is not None) == 2)
        assert template is not None or y is not None
        
        if times is None:        
            if template is not None:
                times=template.times
            else:
                if duration is None:
                    duration = (samples+1)/rate
                if samples is None:
                    samples=max([1,int(math.ceil(duration*rate))])
                if rate is None:
                    rate=samples/duration
                times=numpy.linspace(delay+tjust/rate,delay+duration+tjust/rate,num=samples,endpoint=False)
        
        if y is None:
            y=template.y
                
        return cls(numpy.zeros((len(y),len(times))),y=y,times=times,tjust=tjust)
                
    def pad(self,start=0,end=None, with_=0):
        """
        Pad the TimeMatrix
        
        start - float
            time in seconds at which the zero padded TimeMatrix beggins
        end - float
            time in seconds at which the zero padded TimeMatrix ends
        with_ - float
            value to use for padding
            
        returns a padded version of self
        """
        if end is None:
            end = self.end
        times=self.times
        start_idx=int(math.floor(self.rate*(start-self.start)))
        end_idx=int(math.floor(self.rate*(end-self.start)))
        if start_idx < 0:
            pre=numpy.linspace(self.start+start_idx/self.rate+self.tjust/self.rate,self.start+self.tjust/self.rate,-start_idx,endpoint=False)    
            times=numpy.append(pre,times)
        if end_idx > self.samples:
            post=numpy.linspace(self.end+self.tjust/self.rate,self.end+(end_idx-self.samples)/self.rate+self.tjust/self.rate,(end_idx-self.samples),endpoint=False)    
            times=numpy.append(times,post)
        z=type(self).zeros(times=times,y=self.y)
        z[:]=with_
        self_i0=numpy.abs(z.times-self.start).argmin()
        z[:,self_i0:(self_i0+self.samples)]=self[:,:]
        return z
        
    def zero_pad(self,start=0,end=None):
        """
        Zero pad the time matrix
        """
        return self.pad(start=start,end=end,with_=0)

    def nan_pad(self,start=0,end=None):
        """
        Pad the time matrix with Not a Number (nan)
        """
        return self.pad(start=start,end=end,with_=numpy.nan)
        
    def __add__(self,other):
        if isinstance(other,type(self)):
            #cheking rates are equivalent
            assert abs(self.rate - other.rate) <= self.rate/1000
            assert len(self.y) == len(other.y)
            #assert numpy.all(abs(self.y - other.y) <= self.y/1000) #Spe specific assertion    

            #creating host matrix
            z=self.zero_pad(min([self.start,other.start]),max([self.end,other.end]))
            other_i0=numpy.abs(z.times-other.start).argmin()
            z[:,other_i0:(other_i0+other.samples)]+=other[:,:]
            return z
        else:
            return super().__add__(other)
        
    def __sub__(self,other):
        if isinstance(other,type(self)):
            #cheking rates are equivalent
            assert abs(self.rate - other.rate) <=  self.rate/1000
            assert len(self.y) == len(other.y)
            #assert numpy.all(abs(self.y - other.y) <= self.y/1000) #Spe specific assertion  

            #creating host matrix
            z=self.zero_pad(start=min([self.start,other.start]),end=max([self.end,other.end]))
            other_i0=numpy.abs(z.times-other.start).argmin()
            z[:,other_i0:(other_i0+other.samples)]-=other[:,:]
            return z
        else:
            return super().__sub__(other)
        
    def nan_remove(self,axis=1):
        """
        remove rows (or columns) with nans from ensemble
        
        axis - int
            0 removes columns, 1 removes rows (default)
            
        return a new TimeMatrix with the nans removed
        """
        if axis==1: #removing rows
            nanrows=numpy.isnan(self.array).any(axis=1)
            output = type(self)(self.array[~nanrows,:],y=self.y[~nanrows],times=self.times,tjust=self.tjust)
            if self.db is not None:
                output.db = self.db.ix[~nanrows]
            return output                        
        elif axis==0:
            raise Error("axis=0 not implemented yet")
        else:
            raise Error("invaid value for axis")
        
class Ens(TimeMatrix):
    """
    Object representing an ensemble of Wavs or Envs
    
    Inherits from TimeMatrix
    
    uaid : ndarray
        Unique annotation id.
    times : ndarray
        Array of segment times.   
    annot_type : str
        annotation type from which the ensamble was constructed
    """
    
    @classmethod
    def create(cls, values, times=None, db=None, tjust=0.5, fill_value=None):
        """
        Creates an esemble from a list (or iterable) of Wavs
        
        values - list or iterable of Wavs
            Contains the rows of the ensemble. If iterable should have the len method
        times - numpy array of dimension one 
            defines the times of the object. If None times is inferred from values
        db - pandas DataFrame
            information of each row of the ensemble
        tjust - float
        fill_value - float

        Returns an ensemble (Ens) object
        """

        if times is None:
            raise Error("Infer of times not implemented yet")
        
        start = numpy.amin(times)
        end = numpy.amax(times)
                    
        ens_matrix=numpy.matrix([value.nan_pad(start,end).interpolate(times,fill_value=fill_value) \
                                 for value in values])
        output=cls(ens_matrix, numpy.arange(len(values)), times, tjust=tjust)

        if db is not None:
            output.db=db
        else:
            warnings.warn("Calls to Ens.create should pass a db argument. Will raise Error in the future")
        
        return output

        
    @classmethod
    def hstack(cls, tup, overlap=False):
        """
        Stack ensembles in sequence horizontally (column wise).
        
        tup - sequence of ensembles
            ensembles to be stacked together
        overlap - bool
            determines if overlaps should be allowed 
            If False raises an error if consecutive elements overlap for more
            than a sample. If True applies a liner mixing in the overlapping
            segment.
        """
        if len(tup)==0:
            raise Error("No elements to stack")
        rate = tup[0].rate
        y_N= tup[0].shape[0]
        tjust = tup[0].tjust
        y = tup[0].y
        y_class = tup[0].y.__class__.__name__
        times=tup[0].times
        if not all(abs(ens.rate-rate) <= rate/1000 for ens in tup):
            raise Error("all ensembles in tup should have the same rate")
        if not all(ens.shape[0] == y_N for ens in tup):
            raise Error("all ensembles in tup should have the same number of elements (rows)")
        if not all(ens.tjust == tjust for ens in tup):
            raise Error("all ensembles in tup should have the same tjust")
        if not all(ens.y.__class__.__name__ == y_class for ens in tup):
            raise Error("all ensembles in tup should have the same type for the 'y' component")
        if not isinstance(times,numpy.ndarray):
            raise Error("times should be a numpy.ndarray")

        hpart=[tup[0]] #horizontal partitions (non overlapping Ens)
        for i in range(1,len(tup)):
            if hpart[-1].end > tup[i].start + 1/rate: #overlap
                if not overlap:
                    raise Error("Ensembles overlap in time but overlap=False")
                ovlp1=hpart[-1].crop(tup[i].start,hpart[-1].end)
                ovlp2=tup[i].crop(tup[i].start,samples=ovlp1.samples)
                hpart[-1]=hpart[-1].crop(hpart[-1].start-1/rate,samples=hpart[-1].samples-ovlp1.samples)
                mixv=numpy.linspace(1/ovlp1.samples,1,num=ovlp1.samples,endpoint=False)
                ovlp2.times=ovlp1.times
                hpart.append(ovlp1*(1-mixv) + ovlp2*mixv)
                tupis=tup[i].samples-ovlp1.samples
                hpart.append(tup[i].crop(end=tup[i].end,samples=tupis))
                times=numpy.hstack((times,times[-1]+numpy.arange(1,tupis+1)/rate))
                
            else: #no overlap
                hpart.append(tup[i])        
                times=numpy.hstack((times,times[-1]+numpy.arange(1,tup[i].samples+1)/rate))
        
        #output=Spe.Ens(numpy.hstack(ens.array for ens in hpart), y, times, tjust=tjust)        
        output = cls(numpy.hstack(ens.array for ens in hpart), y, times, tjust=tjust)        
        return output
        
        
    @classmethod
    def vstack(cls, tup):
        """
        Stack ensembles in sequence vertically (row wise).
        
        tup - sequence of ensembles
            ensembles to be stacked together
        """
        if len(tup)==0:
            raise Error("No elements to stack")
        rate = tup[0].rate
        samples = tup[0].samples
        start = tup[0].start
        tjust = tup[0].tjust
        times = tup[0].times
        y_class = tup[0].y.__class__.__name__
        if not all(abs(ens.rate-rate) <= rate/1000 for ens in tup):
            raise Error("all ensembles in tup should have the same rate")
        if not all(ens.samples == samples for ens in tup):
            raise Error("all ensembles in tup should have the same number of samples")
        if not all(abs(ens.start - start) <= (1/rate)/1000 for ens in tup):
            raise Error("all ensembles in tup should have the start time")
        if not all(ens.tjust == tjust for ens in tup):
            raise Error("all ensembles in tup should have the same tjust")
        if not all(ens.y.__class__.__name__ == y_class for ens in tup):
            raise Error("all ensembles in tup should have the same type for the 'y' component")

        y=tup[0].y
        for i in range(1,len(tup)):
            y = y.append(tup[i].y)
        output = cls(numpy.vstack(ens.array for ens in tup), y, times, tjust=tjust)        
        output.db=pandas.concat((ens.db for ens in tup if ens.db is not None),axis=0,ignore_index=True)
        return output
            
    @property
    def uaid(self):
        return self.y   
    
    def average(self):
        """
        Calculates average time trace of the ensamble
        
        returns a Env object
        """        
        return Env(numpy.average(self.array,axis=0), self.rate, delay=self.delay, tjust=self.tjust)
        
    def mean(self):
        """
        Calculates mean time trace of the ensamble
        
        returns a Env object
        """        
        return Env(numpy.nanmean(self.array,axis=0), self.rate, delay=self.delay, tjust=self.tjust)
    
    
    def median(self):
        """
        Calculates median time trace of the ensamble
        
        returns a Env object
        """        
        return Env(numpy.nanmedian(self.array,axis=0), self.rate, delay=self.delay, tjust=self.tjust)
    
    def std(self):
        """
        Calculates Standard Deviation time trace of the ensamble
        
        returns a Env object
        """        
        return Env(numpy.nanstd(self.array,axis=0), self.rate, delay=self.delay, tjust=self.tjust)
        
    def var(self):
        """
        Calculates Variance time trace of the ensamble
        
        returns a Env object
        """        
        return Env(numpy.nanvar(self.array,axis=0), self.rate, delay=self.delay, tjust=self.tjust)
        
    def cv(self):
        """
        Calculates the Coefficient of variation time trace of the ensamble
        
        returns a Env object
        """        
        return Env(numpy.nanstd(self.array,axis=0)/numpy.nanmean(self.array,axis=0), self.rate, delay=self.delay, tjust=self.tjust)

    def eta2(self):
        """
        Calculates the eta square (variance divided by mean square) time trace of the ensamble
        
        returns a Env object
        """        
        return Env(numpy.nanvar(self.array,axis=0)/numpy.square(numpy.nanmean(self.array,axis=0)), self.rate, delay=self.delay, tjust=self.tjust)
        
    def normalize(self,method='meanstd',alpha=1,beta=1):
        """
        normalize the ensemble
        
        method - str
            method used for normalization: 'meanstd', 'alphabeta'
        alpha - float
            used for alphabeta method
        beta - float
            used for alphabeta method

        returns a normalized wav object
        """
        if method == 'meanstd':
            return (self-numpy.mean(self.array,axis=1)[:,numpy.newaxis])/\
                numpy.std(self.array,axis=1)[:,numpy.newaxis]
        elif method == 'alphabeta':
            return (self-alpha*numpy.mean(self.array,axis=1)[:,numpy.newaxis])/\
                numpy.power(numpy.std(self.array,axis=1)[:,numpy.newaxis],beta)
        else:
            raise Error("unknown normalization method")

        
    def to_dataframe(self,channel='ensemble',db_cols=None,**kwargs):
        """
        Returns a data frame representation of the object
        
        channel - string
            name of the channel. Defaults to 'ensemble
        db_cols - list of strings
            names of columns from the annotatin db from which the ensemble was
            constructed, to include in the data frame
        kwargs - dict
            columns to be appended to the dataframe
            
        returns a pandas.DataFrame
        """
        df_l=list()
        for i in range(self.shape[0]):
            df = pandas.DataFrame({'time':self.times})
            df[channel]=self.array[i,:]
            df.insert(0,'annot_id',self.y[i][1])
            df.insert(0,'record_id',self.y[i][0])
            df_l.append(df)
        df=pandas.concat(df_l)
        df['uaid']=(df.record_id*100+df.annot_id).astype(str)
        for col in kwargs:
            df[col]=kwargs[col]
        if db_cols is not None:
            df=df.join(self.db.ix[:,db_cols],on=['record_id','annot_id'])
        return df

    def sample(self, N=1, replace=False):
        """
        Returns a random sample of the ensable
        
        N - integer
            number of samples to retrieve

        replace - boolean
            should items sample be replaced

        returns a new Ens obj with a subset of the rows
        """
        if N > self.shape[0]:
            N=self.shape[0]
        #elements=(numpy.random.sample(N)*self.shape[0]).astype(int)
        elements=numpy.random.choice(self.shape[0],size=N,replace=replace)
        output = type(self)(self.array[elements,:],y=self.y[elements],times=self.times,tjust=self.tjust)
        if self.db is not None:
            output.db = self.db.ix[elements]
        return output   
     
    def bootstrap_resample(self):
        """
        Create a bootstrap resample from the ensemble

        returns a new Ens, the same shape of self, resampled with replacement from self
        """
        return self.sample(N=self.shape[0],replace=True)


    def subset(self, select=None, **kwargs):
        """
        select - string or bool series 
            If it is a string it should evaluate to a boolean in the conntext of the 'by' DataFrame
            If it is a series it should be of length appropiate for 'by' DataFrame
            If None all registers in the 'by' DataFrame will be selected
        **kwargs
            Further arguments for pandas.DataFrame.query
        """
        if self.db is None:
            raise Error("Can't subset if self.db is None")

        if select is not None:
            if type(select)==str: 
                sdb=self.db.query(select, **kwargs).copy()
            else:
                sdb=self.db.ix[select,:].copy()
                
        sa=self.db.index.isin(sdb.index)
        output = type(self)(self.array[sa,:],y=self.y[sa],times=self.times,tjust=self.tjust)
        output.db = sdb
        return output         
        
    def plot(self,tlim=None,*args,**kwargs):
        """
        Plot ensemble vs time using matplotlib.pyplot.plot
        """
        for i in range(self.array.shape[0]):
            matplotlib.pyplot.plot(self.times,self.array[i,:],*args,**kwargs)
        if tlim is not None: 
            matplotlib.pyplot.xlim(tlim)
        matplotlib.pyplot.xlabel("time (s)")
        return self
        
    def get_env(self, key):
        """
        Get element of ensenble by positinal index
        """
        return Env(self.array[key,:], self.rate, delay=self.delay, tjust=self.tjust)
    
    def dejitter(self, tlim=0.05, maxIter=20):
        """
        apply lateral displacements to de-jitter the signal against the median
        """
        c=0
        sas=1
        while sas > 0 and c < maxIter:
            sas=0
            m = self.median()
            for i in range(self.shape[0]):
                tmi=self.get_env(i).correlate(m,mode='full').crop(-0.05,0.05).time_max()
                s=-int(round(tmi*self.rate))
                sas = sas + abs(s)
                self[i,:] = numpy.roll(self.array[i,:],shift=s,axis=0)
            c=c+1
            
        return self            
    
        
class Spe(TimeMatrix):
    """
    Object representing a spectrogram of a Wav
    
    Inherits from TimeMatrix.
        
    freqs : ndarray
        Array of sample frequencies.
    times : ndarray
        Array of segment times.
    method : string
        Method name used to calculate the spectrogram
    method_args: dict
        arguments used for calculating the specgram
    """
    @property
    def freqs(self):
        return self.y
       
    def plot(self, cmap=None, flim=(0,11000), tlim=None, dynamic_range=70, **kwargs):
        """
        Plot the spectogram using matplotlib library
        
        cmap - color map
            color map used to represent the amplitude in dB. 
            If None defults to matplotlib.pyplot.get_cmap('Greys')
        flim - 2-tuple of floats
            frecuency range to plot
        tlim - 2-tuple of floats
            time range to plot
        dynamic_range - float
            dynamic range in dB to plot
        kwargs - dict
            further arguments for matplotlib.pyplot.pcolormesh
        """
        
        if cmap is None: cmap=matplotlib.pyplot.get_cmap('Greys')

        if flim is None: 
            flim_idx=(0,len(self.freqs)-1)
            flim=(self.freqs[0],self.freqs[flim_idx[1]])
        else:
            assert len(flim)==2
            flim_idx = (numpy.argmin(numpy.abs(flim[0] - self.freqs)),numpy.argmin(numpy.abs(flim[1] - self.freqs)))
       
        if tlim is None: 
            tlim_idx=(0,len(self.times)-1)
            tlim=(self.start,self.end)
        else:
            assert len(tlim)==2
            tlim_idx = (numpy.argmin(numpy.abs(tlim[0] - self.times)),numpy.argmin(numpy.abs(tlim[1] - self.times)))

        Z = 10. * self[flim_idx[0]:flim_idx[1],tlim_idx[0]:tlim_idx[1]]
            
        if dynamic_range is not None: #note that the Spe object stores log10 values
            Z=numpy.clip(Z,a_min=float(numpy.amax(Z)-abs(dynamic_range)),a_max=numpy.amax(Z))
            
        ax=matplotlib.pyplot.pcolormesh(self.times[tlim_idx[0]:tlim_idx[1]]+self.tjust/self.rate, self.freqs[flim_idx[0]:flim_idx[1]], \
                        Z, cmap=cmap, rasterized=True, **kwargs).axes
        ax.axis([tlim[0], tlim[1], flim[0], flim[1]])       
        
        ax.set_ylabel('Frequency (kHz)')
        y_ticks=numpy.arange((flim[0]/1000)//1,(flim[1]/1000)//1,2).astype(int)
        y_ticks_labels=y_ticks.astype(str)
        if y_ticks[0]==0:
            y_ticks_labels[0]=''
        ax.get_yaxis().set_ticks(y_ticks*1000)
        ax.get_yaxis().set_ticklabels(y_ticks_labels)
       
    def correlate(self, template, mode='valid', flim=(0,11000)):
        """
        Calculate the correlation with a template
        
        template - Spe
            object with which to correlate
            
        mode - string indicating mode of correlation. 'valid', 'full' or 'same' (see scipy.signal.correlate)
            
        returns a Cor of (T-T_sample) with correspondent correlation coefficient
        """

        #FIXME: this algorithm has numericall issues
        F = template.shape[0]    
        T1 = template.shape[1]    
        if self.shape[0] != F:
            raise Error("self and template should have same number of freqs")

        if flim is None: 
            flim_idx=(0,F-1)
            flim=(self.freqs[0],self.freqs[flim_idx[1]])
        else:
            if len(flim)!=2:
                raise Error("flim should be a 2-tuple")
            flim_idx = (numpy.argmin(numpy.abs(flim[0] - self.freqs)),numpy.argmin(numpy.abs(flim[1] - self.freqs)))
        F = flim_idx[1]-flim_idx[0]
            
        X1 = template[flim_idx[0]:flim_idx[1],:].array
        X2 = self[flim_idx[0]:flim_idx[1],:].array

        if not (T1<=self.shape[1] or mode!='valid'):
            raise Error("template should be shorter than self if mode is 'valid'")
        if not self.method == template.method:
            raise Error("self and template have to be calculated with the same method")
        
        X1=(X1-numpy.mean(X1))/numpy.std(X1)
        U=numpy.ones((F,T1))
        r=scipy.signal.correlate2d(X1,X2,mode=mode)/ \
            numpy.sqrt(F*T1*scipy.signal.correlate2d(U,numpy.square(X2),mode=mode)-numpy.square(scipy.signal.correlate2d(U,X2,mode=mode)))
        #FIXME: substracting two large numbers and the calculatng sqrt. Numerical issues. 
        if mode == 'full':
            delay=-template.duration
        else:
            delay=0
        return Cor(r[0,::-1],self.rate,delay=delay)    
        
class CorSpe(TimeMatrix):
    """
    Object representing a correlation spectrogram 
    (Correlation coefficient at different times and delays)
    
    Inherits from Spe.
        
    taus : ndarray
        Array of correlation time displacement (taus or 'delays').
    times : ndarray
        Array of segment times.
    method : string
        Method name used to calculate the spectrogram
    method_args: dict
        arguments used for calculating the specgram
    """  
    @property
    def taus(self):
        return self.y
        
    def plot(self, cmap=None, norm=(0.5,1), **kwargs):
        """
        Plot the cross-correlation spectogram using matplotlib library
        
        cmap - color map
            color map used to represent the amplitude in dB. 
            If None defults to matplotlib.pyplot.get_cmap('Greys')
        norm - 2-tuple of floats
            lower and upper limit for the colormap
        """
        if cmap is None: cmap=matplotlib.pyplot.get_cmap('Greys')
        
        matplotlib.pyplot.pcolormesh(self.times, self.taus, \
                        numpy.ma.masked_invalid(self.array), \
                        norm=ClipNormalize(vmin=norm[0],vmax=norm[1]), \
                        cmap=cmap, rasterized=True, **kwargs)
        matplotlib.pyplot.axis([self.start, self.end, self.taus[0], self.taus[-1]])       
        
class ClipNormalize(matplotlib.colors.Normalize):
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vmax], [0, 1]
        return numpy.ma.masked_array(numpy.interp(value, x, y))        
        
from Wav import Cor, Env        
