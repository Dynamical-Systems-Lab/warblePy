#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rec.py module of WarblePy. 

Rec: represents a record, that is a collection of several Wavs recorded simultaneously.
     record usually contain sound and EMG data. 
Ann: Inherits from Rec. Represents an annoation within a record. 

@author: Alan Bush
"""

import collections
import warnings
import matplotlib.pyplot
import scipy.signal
import copy
import pandas
import os
import re
from Wav import Wav, Cor, Promise

class Error(Exception): pass

class Rec(collections.abc.MutableMapping):
    '''
    Object representing the recording of one or several channels
    
    If lazy loading, the record consists of WavPromis instead of Wavs. 
    The promis are fulfil on demand. Any pre-processing is done at that point by the promise call.
    '''
    def __init__(self,record=None,annot=None,**kwargs):
        super().__setattr__('channels',kwargs)
        if record is None: record=dict()
        super().__setattr__('record',record)
        super().__setattr__('annot',annot)
    
    def __setitem__(self, key, value):
        self.channels[key] = value
    
    def __getitem__(self, key):
        if isinstance(self.channels[key],Promise):
            self.channels[key]=self.channels[key]()
        return self.channels[key]
    
    def __delitem__(self, key):
        del self.channels[key]
    
    def __iter__(self):
        self.fulfil()
        return iter(self.channels)
    
    def __len__(self):
        return len(self.channels)
    
    def __repr__(self):
        s=str(self.__class__.__name__) + "("
        if not self.rid is None: s+="record_id=%i, " % self.rid    
        s+="channel(s): " 
        for key in self.channels.keys(): 
            s+=str(key)
            if isinstance(self.channels[key],Promise):
                s+="*"
            s+=", "
        s=s[0:-2]+")"
        return s

    def __getattr__(self, attr):
        if attr in self.channels:
            if isinstance(self.channels[attr],Promise):
                self.channels[attr]=self.channels[attr]()
            return self.channels[attr] 
        if attr in self.record:
            return self.record[attr]
        else: 
            return super().get(attr)

    def __setattr__(self, attr, value):
        if attr in self.channels:
            self.__setitem__(attr, value)
        if attr in self.record:
            self.record[attr]=value
        else:
            super().__setattr__(attr,value)
    
    def __delattr__(self, attr):
        if attr in self.channels:
           del self.channels[attr] 
        if attr in self.record:
            warnings.warn(attr + " in Rec.record, use: del Rec.record["+attr+"]")
        else:
            super().__delitem__(attr)       
    
    def write(self,name,path=None,folder=True):
        """
        Writes the TimeMatrix object to disk
        
        name - str
            base name of the file/fodler (no extension)
        
        path - str
            where to write the file. If None the current directory is used.
            
        folder - bool
            should a folder named 'name' be created
        """        
        basename=name
        if path is None:
            path = os.getcwd()
        name = os.path.join(path,name)
        if folder is True:
            if not os.path.isdir(name):
                os.mkdir(name)
            name = os.path.join(name,basename)
        
        if self.record:
            pandas.Series(self.record).to_csv(name+"_record.dat",sep='\t',index=True,na_rep='NA')
        if self.annot is not None:
            for annot_type in self.annot:
                self.annot[annot_type].to_csv(name+"_annot["+annot_type+"].dat",sep='\t',index=True,na_rep='NA')
        
        for ch in self.channels:
            self.channels[ch].write(name+"["+ch+"].wav")
                
                
    @classmethod
    def read(cls,name,path=None,folder=True):
        """
        Reads a TimeMatrix object from disk
        
        name - str
            base name of the file/folder (no extension)
        
        path - str
            where to read the file from. If None the current directory is used.
            
        folder - bool
            are the files in a folder named 'name'?
        """
        basename=name
        if path is None:
            path = os.getcwd()
        if folder is True:
            path = os.path.join(path,basename)
            if not os.path.isdir(path):
                raise Error("no folder named "+path+"'")
            
        record_dat = os.path.join(path,name) + "_record.dat"
        if os.path.isfile(record_dat):
            try:
                record=pandas.read_csv(record_dat,squeeze=True,index_col=0,\
                                   header=None,parse_dates=True,sep='\t',engine='python')\
                                   .to_dict()
            except:
                record=dict()
        else:
            record=dict()
        
        ch_regex = re.compile("^"+name+"\\[(.*)]\\.wav$")
        ch_list = [ch_regex.search(ch_file).group(1)\
                   for ch_file in filter(ch_regex.match, os.listdir(path))]        
        rec_channels={ch:Wav.read(os.path.join(path,name)+"["+ch+"].wav") \
                        for ch in ch_list}
                        
        annot_regex = re.compile("^"+name+"_annot\\[(.*)]\\.dat$")
        annot_list = [annot_regex.search(annot_file).group(1)\
                   for annot_file in filter(annot_regex.match, os.listdir(path))]        
        
        annot=dict()
        for annot_type in annot_list:
            annot_db=pandas.read_table(os.path.join(path,name)+"_annot["+annot_type+"].dat")
            if "annot_id" in annot_db.columns:
                annot_db.set_index(['annot_id'],inplace=True,drop=False)
            else:
                raise Error(annot_type+" has nor 'record_id' column") 
            annot[annot_type]=annot_db.copy()
        
        return cls(record=record,annot=annot,**rec_channels)
        
    def copy(self,*args,**kwargs):
        """
        returns a copy of the Rec
        """   
        return Rec(record=self.record.copy(),annot=self.annot.copy(),**{w.copy(*args,**kwargs) for w in self})   
            
    def fulfil(self):
        for attr in self.channels:
            if isinstance(self.channels[attr],Promise):
                self.channels[attr]=self.channels[attr]()
        return self

    @property
    def rid(self):
        """
        Returns the record id
        """
        if (self.record is None) or (not 'record_id' in self.record):
            return None
        else:
            return int(self.record['record_id'])

    def plot(self,tlim=None,*args,**kwargs):
        """
        plots every channel of the record
        
        tlim - 2-tuple of floats or None
            time limits for the plot
            if None, it defaults to the min start and max end of all the channels
        args and kwargs are passed to plt.plot
        """
        self.fulfil()
        if tlim is None:
            tlim=(min([self.channels[ch].start for ch in self.channels]),\
                  max([self.channels[ch].end for ch in self.channels]))
        i=1
        for ch in self.channels:
            matplotlib.pyplot.subplot(len(self.channels),1,i)
            self.channels[ch].plot(tlim=tlim,*args,**kwargs)
            matplotlib.pyplot.ylabel(ch)
            i+=1
        return self
            
    def crop(self,start=None,end=None,aligned=True):
        """
        Returns a cropped copy of the record
        
        start: float
            start time in seconds
        end: float
            end time in seconds
        aligned, bool
            indicates if cropped Rec should be time aligned with original Rec.
            if False delay is set to zero. 
        """
        rec_channels={}
        for ch in self.channels:
            ch_wav=self.channels[ch].crop(start=start,end=end,aligned=aligned)
            if ch_wav is not None:
                rec_channels[ch]=ch_wav
        return Rec(record=self.record,annot=self.annot,**rec_channels)
    
    def zero_pad(self,start=None,end=None):
        """
        Zero pad all channels of the record.
        """
        if start is None:
            start = min(self[ch].start for ch in self)            
        if end is None:
            end = max(self[ch].end for ch in self)
        for ch in self:
            self[ch]=self[ch].zero_pad(start,end)            
        return self
        
    def set_delay(self,delay,channel=None):
        """
        sets the delay of the Rec.
        
        delay - float
            the new delay 
        channel - str or None
            if str, the channel name to be set to delay
            if None, the channel of min delay is set to delay
            
        Note that the relative differences between the channels are mantained
        The annotations are updated for consistency
        """
        ch_delays={ch:self.channels[ch].delay for ch in self.channels}
        if channel is None:
            channel=min(ch_delays,key=ch_delays.get)
        delta_delay=delay-ch_delays[channel]
        for ch in self.channels:
            self.channels[ch]=self.channels[ch].set_delay(ch_delays[ch]+delta_delay)
            
        for ann_db in self.annot:
            self.annot[ann_db]['start']+=delta_delay
            self.annot[ann_db]['end']+=delta_delay
            
        return self       

    def shift(self,*args,**kwargs):
        """
        returns a time shifted copy of the Rec
        
        delay - float
            new delay
        start - float
            new start
        end - float
            new end
        by - float 
            time in seconds by which to shif the Rec relative to current time
        """   
        return Rec(record=self.record.copy(),annot=self.annot.copy(),**{w.shift(*args,**kwargs) for w in self})   
        
        
    def set_channel(self,channel,wav):
        """
        Sets a new channel of the record and returns the modified Rec
        
        channel - str
            indicates the name of the channel
        wav - Wav
            the Wav object of the new channel
        """
        self[channel]=wav
        return self
        
    def get_annot(self,annot,annot_id=None,select=None,annot_type=None,flanking=0):
        """
        get a croped version of the record according to the annotation data
        
        annot - str or pandas.Series
            pandas.Series with elements start and stop. usually a row of a annotation DataFrame
            if string, should be a key name of self.annot and annot_id should be given
        annot_id - int
            annotation id, within the specified element of self.annot
            if annot_id=='any', the first annotation is returned
        select - str of pandas.Series of booleans
            selection criteria of a single row of the annot DataFrame. 
            Used if 'annot' is a str and 'annot_id' is None
        annot_type - str
            descripction of the annotation type. Used if annot is a pandas.Series
            If annot is a str, annot_type is set to annot
        flanking - float or 2-tuple of floats
            time in seconds by which to flank the annotation
            if 2-tuple, the first element extends the annot by this amount at the begging
            and the second element extends the annot at the end. Negative values reduce
            the annot.
        
        returns a Ann object
        """
        #rec_channels={ch:self.channels[ch].crop(start=annot.start-flanking,end=annot.end+flanking,aligned=True) for ch in self.channels}    
        #return Ann(record=self.record,annot=annot,**rec_channels)  
        if isinstance(annot,str):
            if annot not in self.annot:
                raise Error("'annot' should be a key in self.annot")
                
            if select is not None:
                if isinstance(select,str):
                    sel_db=self.annot[annot].copy().query(select)        
                elif isinstance(select,pandas.Series):
                    sel_db=self.annot[annot].copy().ix[select,:]        
                else:
                    raise Error("Invalid type for select")
            else:
                sel_db=self.annot[annot].copy()

            if len(sel_db)==0:
                raise Error("No annotations match your selection")
                    
            if annot_id is None or annot_id=='any':
                if len(sel_db)>1 and annot_id is None:
                    warnings.warn(str(len(sel_db)) + " annotations match your selection. Using first.")
                annot_id=sel_db.iloc[0].annot_id
                
            if annot_id not in set(self.annot[annot].annot_id):
                raise Error("annot_id="+str(annot_id)+" not present in self.annot[annot]")
            annot_type=annot
            annot_info=self.annot[annot].loc[annot_id].copy()
        elif isinstance(annot,pandas.Series):
            annot_info=annot.copy()
        else:
            raise Error("annot should be a string or pandas.Series")

        annot={i:self.annot[i].copy().ix[(self.annot[i].start<annot_info.end) & (self.annot[i].end>annot_info.start)]\
                   for i in self.annot}
           
        if not isinstance(flanking, tuple):
            flanking = (flanking, flanking)    
            
        annot_info.start=annot_info.start-flanking[0]
        annot_info.end=annot_info.end+flanking[1]
        try:
            rec_channels=self.crop(start=annot_info.start,end=annot_info.end,aligned=True).channels
        except:
            raise Error("record "+str(self.rid)+" failed to retrive annotation "+str(annot_type)+" "+str(annot_info.annot_id)+" ["+str(annot_info.start)+";"+str(annot_info.end)+"]")
        return Ann(record=self.record,annot_info=annot_info,annot_type=annot_type,annot=annot,**rec_channels)        

    def iter_annot(self,annot_type,select=None,flanking=0, **kwargs):
        """
        Creates an iterator over annotations
        
        annot_type - str
            name of the annotation dataframe in rec.annot
        select - str or boolean pandas.Series
            filter over the dataframe before the iterator is created
        flanking - float
            time in seconds by which to flank the annotation
        kwargs - dict
            further arguments for pandas.DataFrame.query if select is a str
            
        returns an iterator over annotations
        """
        if not annot_type in self.annot:
            raise Error(str(annot_type)+" not in rec.annot")
        db=copy.deepcopy(self.annot[annot_type])    
        #filtering
        if select is not None:
            if type(select)==str: 
                db.query(select,inplace=True, **kwargs)
            else:
                db=db.ix[select,:]
        return ann_iter(self,db,annot_type,flanking)
        
    def to_dataframe(self,times=None,**kwargs):
        """
        Returns a data frame representation of the record
        
        times - str, numeric vector or none
            if str, corresponds to the channel used to determine the time points returned
            if numeric vector, the time points to return
            if None, defaults to the time points of the channel of minimum rate
        kwargs - dict
            columns to be appended to the dataframe
            
        returns a pandas.DataFrame
        """
        if times is None:
            rates_dict={ch:self[ch].rate for ch in self}
            times=min(rates_dict,key=rates_dict.get)
        if isinstance(times,str):
            if not times in self:
                raise Error('if times is a str, it should be a channel of the record')
            times = self[times].get_times()

        df = pandas.DataFrame({'time':times})
        for ch in self:
            df[ch]=self[ch].interpolate(times)
        for col in kwargs:
            df[col]=kwargs[col]
        return df
        
class Ann(Rec): 
    """
    Object represeting a annotatios from a record
    
    Inherits from Rec. Additional properties:
    annot: pandas series of annotation db
    """
    def __init__(self,record=None,annot=None,annot_info=None,annot_type=None,**kwargs):
        super().__init__(record,annot,**kwargs)
        self.annot_type=annot_type
        self.annot_info=annot_info
        if annot_info is not None:
            self.aid=annot_info.annot_id  
        else:
            self.aid=None
            
    def __repr__(self):
        s=str(self.__class__.__name__) + "("
        if not self.rid is None: s+="record_id=%i, " % self.rid    
        if not self.aid is None: s+="annot_id=%i, " % self.aid   
        if not self.annot_type is None: s+="type=%s, " % self.annot_type   
        if not self.annot_info is None: s+="[%.2f,%.2f], " % (self.annot_info.start, self.annot_info.end)
        s+="channel(s): " 
        for key in self.channels.keys(): 
            s+=str(key)
            if isinstance(self.channels[key],Promise):
                s+="*"
            s+=", "
        s=s[0:-2]+")"
        return s 

    @property
    def info(self):
        return self.annot_info
        
class ann_iter(collections.abc.Iterator):
    def __init__(self,obj,annot_db,annot_type,flanking=0):
        self.obj=obj
        self.annot_db=annot_db
        self.annot_type=annot_type
        self.flanking=flanking
        self.idx=0
    def __iter__(self):
        return(self)
    def __next__(self):
        if self.idx<len(self.annot_db):
            ann=self.obj.get_annot(self.annot_db.iloc[self.idx],annot_type=self.annot_type,flanking=self.flanking)
            self.idx+=1
            return ann
        else:
            raise StopIteration
    def __len__(self):
        return len(self.annot_db)
        
        