#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:17:46 2016

@author: alan
"""

#cargo paquetes estandar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import os.path
import math
import collections
import warnings

#cargo paquetes de Warble
from Wav import Wav, Env
from Spe import Spe

class Error(Exception): pass


def detect_all(env):
    """
    detects the entire envelope as a ROI

    env - Env to be annotated
        
    returns a pandas DataFrame with columns 'annot_id', 'start', 'end', 'comment'
    """
    col_names = ['start','end','comment']
    annot_db = pd.DataFrame(columns=col_names) 
    annot_db.loc[0] = [env.start,env.end,'']          
    annot_db.insert(0,'annot_id',0)      
    return annot_db

                    
def detect_env(env, threshold, max_ROI=100):
    """
    detects regions of interest in an Env object

    env - Env to be annotated
    threshold - float or 2-tuple of floats
        upper/lower threshold for the segmentation
    max_ROI - int
        maximun number of regions of interest in a record
        
    returns a pandas DataFrame with columns 'annot_id', 'start', 'end', 'comment'
    """
    if not isinstance(threshold,collections.Iterable):
        upper_threshold=threshold
        lower_threshold=threshold
    else:
        upper_threshold=max(threshold)
        lower_threshold=min(threshold)
    
    col_names = ['start','end','max_env','comment']
    annot_db = pd.DataFrame(columns=col_names) 
    zero_cross_ixs = np.where(np.diff(np.signbit(env.array-lower_threshold)))[0]
    reset_value=min(np.amin(env.array),0)

    search_detect = True
    loop_count=0
    while search_detect:
        loop_count+=1
        env_max = np.amax(env.array)       
        env_max_ix = np.nanargmax(env.array)
        if loop_count > max_ROI:
            raise Error("More ROIs detected than max_ROI")
        if env_max > upper_threshold:
            #finding start and end of current ROI
            if len(zero_cross_ixs)==0:
                start = round(env.start,3)
                end = round(env.end,3)
            elif env_max_ix <= zero_cross_ixs[0]:
                start = round(env.start,3)
                end = round(float(env.get_time(zero_cross_ixs[0])+(1-env.tjust)/env.rate),3)
            elif env_max_ix > zero_cross_ixs[-1]:
                start = round(env.get_time(zero_cross_ixs[-1])-env.tjust/env.rate,3)
                end = round(env.end,3)
            else:
                zc_ix = np.where(np.diff(np.signbit(zero_cross_ixs-env_max_ix)))[0]
                start = round(float(env.get_time(zero_cross_ixs[zc_ix])[0]-env.tjust/env.rate),3)
                end = round(float(env.get_time(zero_cross_ixs[zc_ix+1])[0]+(1-env.tjust)/env.rate),3)
            
            overlaps = False
            for i, row in annot_db.iterrows():
                if row.start < end and row.end > start:
                    overlaps = True
            if not overlaps:
                annot_db.loc[len(annot_db)] = [start,end,env_max,'']          
            zidx=env._crop_idx(start,end)
            env[zidx[0]:zidx[1]+1]=reset_value
            #env.plot()
            #env.crop(start,end).plot()
        else:
            search_detect = False

    annot_db.sort_values(by='start',ascending=True,inplace=True)
    annot_db.insert(0,'annot_id',np.arange(len(annot_db)).astype(np.int64))      
    return annot_db

    
def match_obj(obj,template,threshold=0.7,allow_overlap=False,max_ROI=100,scan=None):
    """
    Match a single obj to a template
    
    obj - Wav, Env or Spe object
        The object to be annotated
    template - Wav, Env or Spe object
        The pattern to matched to
    threshold - float
        The minimum correlation value to be assigned as a positive match
    allow_overlap - boolean or float
        indicates if overlaps of the template should be allowed
        if float in [0,1], it indicates the fraction of the motif length that is allowed to overlap
    max_ROI - int
        maximun number of regions of interest in a record
    scan - None, float or 2-tuple of floats
        range to scan. If None, the entire obj is scanned, ignoring any 'delay' of obj.
        If scan is a float, 'obj' is scanned from template.delay-scan to template.delay+scan
        If scan is a 2-tuple, 'obj' is scanned from template.delay+scan[0] to template.delay+scan[1]
        
    returns a pandas DataFrame with columns 'annot_id', 'start', 'end','max_cor', 'comment'
    """
    allow_overlap=float(allow_overlap)
    col_names=['start','end','max_cor','comment']
    annot_db = pd.DataFrame(columns=col_names) 

    #conforming obj to template
    obj=conform(obj,to=template)
    #selecting region of obj to scan
    if scan is not None:
        if not isinstance(scan,tuple):
            scan = (-abs(scan),abs(scan))
        sobj=obj.crop(template.start+scan[0],template.end+scan[1])
    else:
        sobj=obj
    #doing correlation
    cc=sobj.correlate(template).set_delay(sobj.delay)
    #defining regions of correlations as annotations
    search_match=True
    loop_count=0
    while search_match:
        loop_count+=1
        if loop_count > max_ROI:
            raise Error("More ROIs detected than max_ROI")

        cc_max=np.amax(cc.array)        
        cc_tmax=cc.time_max()
       
        if cc_max > threshold:
            overlap=0
            for i, row in annot_db.iterrows():
                if row.start <= cc_tmax+template.duration and row.end > cc_tmax:
                    overlap+=(min(cc_tmax+template.duration,row.end)-max(cc_tmax,row.start))/template.duration
            if overlap <= allow_overlap:
                annot_db.loc[len(annot_db)]=[cc_tmax,cc_tmax+template.duration,cc_max,'']          
                ccidx=cc._crop_idx(cc_tmax-template.duration/2,cc_tmax+template.duration/2)
                cc[ccidx[0]:ccidx[1]+1]=0      
     
            else:
                ccidx=cc._crop_idx(cc_tmax-template.duration/4,cc_tmax+template.duration/4)
                cc[ccidx[0]:ccidx[1]+1]=0
            #cc.plot()
        else:
            search_match = False

    annot_db.sort_values(by='start',ascending=True,inplace=True)
    annot_db.insert(0,'annot_id',np.arange(len(annot_db)))      
    
    return annot_db

    
def conform(obj,to):
    """
    Conforms an object to the type of a template

    obj - Wav, Env or Spe
        The object to be modified to conform to an other object
    to - Wav, Env or Spe
        The object used as a type and method template
    
    returns a Wav, Env or Spe, depending on the type of 'to'    
    """    
    if type(obj).__name__ == type(to).__name__:
        if obj.method == to.method and obj.method_args == to.method_args:
            return obj
        else:
            raise Error("Could not conform obj to template.")
    elif type(obj).__name__=='Wav' and type(to).__name__=='Spe':
        return obj.spectrogram(method=to.method,**to.method_args)
    elif type(obj).__name__=='Wav' and type(to).__name__=='Env':
        return obj.envelope(method=to.method,**to.method_args)
    else:
        raise Error("Could not conform obj to template.")

    
def consolidate(annot_db,when,additive='all'):
    """
    Consolidate annotation rows. Combines together contiguous annotation rows of a record when a criteria is met.
    
    annot_db - pandas DataFrame with required columns 
    when - str or callable
        criteria to define when to consolidate two or more annotations
        The criteria should evaluate to True or False on the candidate consolidated annotation row
        if callable, should accept a pandas.Series and return a boolean
        if str, should be a condition that evaluates to True or False when evaluated on a pandas.Series. 
    additive - list of str or 'all'
        indicates which variables should be treated as 'additive'.
        If 'all', all the variables not required will be treated as additive
        
    returns a pandas DataFrame with columns 'record_id', 'annot_id', 'start', 'end', 
        'comment', 'duration', 'consolidated_duration', the additive variables and all other varibles 
        unmodified with respect to the seed row. 
    """
    cols=['record_id','annot_id','start','end','comment','duration','datetime']

    if set(['start','end']).issubset(annot_db.columns.values) and 'duration' not in annot_db.columns.values:
        annot_db['duration']=annot_db.end-annot_db.start
    if not set(cols).issubset(annot_db.columns.values):
        raise Error("Columns 'record_id', 'annot_id', 'start', 'end', 'comment', 'duration' and 'datetime' required in annot_db")
    if str(additive).lower()=='all':
        additive=list(set(annot_db.columns.values).difference(cols))
    if not isinstance(additive,collections.Iterable):
        raise Error("additive should be a list or 'all'")
    if isinstance(when,str): #when='(c_duration/duration)>0.5'
        when_f = lambda s: eval(when,dict(s))
    else:
        when_f = when

    annot_db[['record_id','annot_id']] = annot_db[['record_id','annot_id']].astype(np.int64)
    annot_db.set_index(['record_id','annot_id'],drop=False,inplace=True,verify_integrity=True)

    def consolidate_record(annot_db):
        cons_db = pd.DataFrame()         
        annot_db=annot_db.copy()
        annot_db.sort_values('start',inplace=True) 
        i=0; j=1;
        while i < len(annot_db)-1:
            if j==1:
                curr_s=annot_db.iloc[i].copy()
                curr_s['consolidated_duration']=curr_s.duration
                merge_s=annot_db.iloc[i].copy()
            merge_s.end=annot_db.iloc[i+j].end
            merge_s.duration=merge_s.end-merge_s.start
            merge_s['consolidated_duration']=curr_s.consolidated_duration+annot_db.iloc[i+j].duration
            merge_s.comment=curr_s.comment+annot_db.iloc[i+j].comment
            merge_s.datetime=curr_s.datetime

            for var in additive:
                #var='vS_energy'
                merge_s[var]=curr_s[var]+annot_db.iloc[i+j][var]
            if when_f(merge_s):
                curr_s = merge_s.copy()
                j+=1
                if i + j >= len(annot_db):
                    cons_db=cons_db.append(curr_s.copy())
                    i=len(annot_db)
            else:
                cons_db=cons_db.append(curr_s.copy())
                i+=j
                j=1
                if i + 1 == len(annot_db): #adding last register
                    curr_s=annot_db.iloc[i].copy()
                    curr_s['consolidated_duration']=curr_s.duration
                    cons_db=cons_db.append(curr_s.copy())
        return cons_db
        
    cons_db=annot_db.groupby('record_id').apply(consolidate_record)

    cols.append('consolidated_duration')
    cols += [c for c in annot_db.columns.values if c not in cols]
    cons_db=cons_db.reindex_axis(cols,axis=1)
    cons_db[['record_id','annot_id']] = cons_db[['record_id','annot_id']].astype(np.int64)
    cons_db.set_index(['record_id','annot_id'],drop=False,inplace=True,verify_integrity=True)
   
    return cons_db

    
def difference(x,y):
    """
    Calculates the difference between annotations DataFrames
    
    x - Pandas DataFrame
    y - Pandas DataFrame
    
    x and y should have columns 'record_id','annot_id','start' and 'end'.
    
    returns a pandas data with annotations corresponding to the difference of 
    annotations on x and y (x-y)
    """
    #rid=20170323183253
    #x=w.annot['QC']
    #y=w.annot['sound1']
    cols=['record_id','annot_id','start','end']

    if hasattr(x, 'name'):
        x_name=x.name
    else:
        x_name='x'

    if not set(cols).issubset(x.columns.values):
        raise Error("Columns '"+"', '".join(cols)+"' required in x DataFrame")
    if not set(cols).issubset(y.columns.values):
        raise Error("Columns '"+"', '".join(cols)+"' required in y DataFrame")
        
    annot_db=pd.DataFrame(columns=cols + ["x_id"])
    for rid in x.record_id:
        if rid in set(y.record_id):
            annot_rec=_difference_(x.ix[rid].copy(),y.ix[rid].copy())
            annot_rec.insert(0,'record_id',int(rid))
            annot_db=annot_db.append(annot_rec, ignore_index=True, verify_integrity=True)    
        else:
            annot_rec=x.ix[rid,cols].copy()
            annot_rec["x_id"]=annot_rec.annot_id
            annot_db=annot_db.append(annot_rec, ignore_index=True, verify_integrity=True)  
            
    annot_db["x_id"] = annot_db["x_id"].astype(int)
    annot_db.rename(index=str, columns={"x_id": x_name + "_annot_id"},inplace=True)
    annot_db[['record_id','annot_id']] = annot_db[['record_id','annot_id']].astype(np.int64)    

    annot_db.sort_values(['record_id','start'],inplace=True) 
    annot_db.set_index(['record_id','annot_id'],drop=False,inplace=True,verify_integrity=True)
    return annot_db
    
def _difference_(x,y):
    #x=w.annot['QC'].query("record_id==20170412113127")    
    #y=w.annot['sound1'].query("record_id==20170412113127")
    #x,y=(x.ix[rid],y.ix[rid])  
    i=0
    j=0
    annot_db = pd.DataFrame(columns=['start','end','x_id']) 
    while i<len(x) and j<len(y):
        x_start=x.start.iloc[i] 
        x_end=x.end.iloc[i]
        x_id=int(x.annot_id.iloc[i]) 
        y_start=y.start.iloc[j]
        y_end=y.end.iloc[j]
        if y_start <= x_start < x_end <= y_end:
            #     xxxxxxxxxxxx
            #  yyyyyyyyyyyyyyyyyy
            i+=1
            
        elif x_start < y_start < y_end < x_end:
            #     xxxxxxxxxxxx
            #        yyyyy  
            annot_db.loc[len(annot_db)]=[x_start,y_start,x_id]
            x.set_value(x.index[i],'start',y_end)
            j+=1

        elif x_start < y_start < x_end <= y_end:
            #     xxxxxxxxxxxx
            #         yyyyyyyyyyyy
            annot_db.loc[len(annot_db)]=[x_start,y_start,x_id]
            i+=1
            
        elif y_start <= x_start < y_end < x_end:
            #     xxxxxxxxxxxx
            # yyyyyyyyy
            x.set_value(x.index[i],'start',y_end)
            j+=1
        
        elif x_end <= y_start:    
            #    xxxxxxxxxxxxx
            #                    yyyyyyyy
            annot_db.loc[len(annot_db)]=[x_start,x_end,x_id]     
            i+=1
            
        elif y_end <= x_start:   
            #       xxxxxxxxxxx
            # yyyy   
            j+=1
            
        else:
            raise Error("Should never get here")
   
    while i<len(x):
        annot_db.loc[len(annot_db)]=[x.start.iloc[i],x.end.iloc[i],int(x.annot_id.iloc[i])]
        i+=1             
                     
    annot_db.sort_values(by='start',ascending=True,inplace=True)
    annot_db.insert(0,'annot_id',np.arange(len(annot_db)))         
    return annot_db    
    
    
    
def intersect(x,y):
    """
    Intersects annotations DataFrames
    
    x - Pandas DataFrame
    y - Pandas DataFrame
    
    x and y should have columns 'record_id','annot_id','start' and 'end'.
    If x and y have 'name' attributes, these are used to define the x/y_annot_id column name
    
    returns a pandas data with annotations corresponding to the intersection of 
    annotations on x and y
    """
    
    cols=['record_id','annot_id','start','end']

    if not set(cols).issubset(x.columns.values):
        raise Error("Columns '"+"', '".join(cols)+"' required in x DataFrame")
    if not set(cols).issubset(y.columns.values):
        raise Error("Columns '"+"', '".join(cols)+"' required in y DataFrame")
        
    if hasattr(x, 'name'):
        x_name=x.name
    else:
        x_name='x'
        
    if hasattr(y, 'name'):
        y_name=y.name
    else:
        y_name='y'

    annot_db=pd.DataFrame(columns=cols+["x_id","y_id"])
    for rid in list(set(x.record_id) & set(y.record_id)):
        #rid=20170411040128
        annot_rec=_intersect_(x.ix[rid],y.ix[rid])
        annot_rec.insert(0,'record_id',int(rid))
        annot_db=annot_db.append(annot_rec, ignore_index=True, verify_integrity=True)    

    annot_db[["x_id","y_id"]] = annot_db[["x_id","y_id"]].astype(int)
    annot_db.rename(index=str, columns={"x_id": x_name + "_annot_id", "y_id": y_name + "_annot_id"},inplace=True)
    annot_db[['record_id','annot_id']] = annot_db[['record_id','annot_id']].astype(np.int64)
    
    annot_db.sort_values(['record_id','start'],inplace=True) 
    annot_db.set_index(['record_id','annot_id'],drop=False,inplace=True,verify_integrity=True)
    return annot_db
    

def _intersect_(x,y):
    #x=wx.annot['pb_xmotif'].query("record_id==20170411001622")    
    #y=wx.annot['vS_activity3'].query("record_id==20170411001622")
    #x,y=(x.ix[rid],y.ix[rid])  
    i=0
    j=0
    annot_db = pd.DataFrame(columns=['start','end','x_id','y_id']) 
    while i<len(x) and j<len(y):
        if x.start.iloc[i]<y.end.iloc[j] and x.end.iloc[i]>y.start.iloc[j]:
            annot_db.loc[len(annot_db)]=[\
                max(x.start.iloc[i],y.start.iloc[j]),\
                min(x.end.iloc[i],y.end.iloc[j]),\
                int(x.annot_id.iloc[i]),\
                int(y.annot_id.iloc[j])]
            if x.end.iloc[i]<y.end.iloc[j]:
                i+=1
            else:
                j+=1
        elif x.end.iloc[i] <= y.start.iloc[j]:
            i+=1
        elif x.start.iloc[i] >= y.end.iloc[j]:
            j+=1
        else:
            raise Error("unsorted input DataFrames")

    annot_db.sort_values(by='start',ascending=True,inplace=True)
    annot_db.insert(0,'annot_id',np.arange(len(annot_db)))         
    return annot_db
    
class Manually:
    """
    Class to manually annotate warble object
    
    use the mouse to select the regions of interest
    enter - to annotate and show next record
    backspace - to clear current selections
    escape - to exit
    other keys are used as comments
    """
    def __init__(self, warble, fname='AnnotateWarble.csv', channel='vS', ylim=(-0.5,0.5), shuffle=True, plot_fun=None):
        self.col_names=['record_id','annot_id','start','end','comment']
        self.shuffle=shuffle
        self.fig = None
        self.warble = warble
        self.start = None
        self.annot_current = pd.DataFrame(columns=self.col_names) 
        self.fname = fname
        self.comment = ''
        self.ylim = ylim
        self.plot_fun=plot_fun
        if os.path.isfile(fname):
            self.annot_db = pd.read_table(fname)
            for col in self.col_names:
                if not col in self.annot_db.columns:
                    raise Error("incorrect columns in fname")
        else:
            self.annot_db = pd.DataFrame(columns=self.col_names) 
    
        self.wav = None
        self.rec = None
        self.channel = channel
        self.begin()

    def get_next_record(self):
        remaining_rid=self.get_remaining_rid()
        if len(remaining_rid) == 0: 
            print("No remaining records")
            self.stop()
            return None
        if self.shuffle: remaining_rid=remaining_rid.sample(frac=1)
        return self.warble.get_record(remaining_rid.iloc[0])
    
    def get_remaining_rid(self):
        remaining_rid=self.warble.record_db.ix[~self.warble.record_db.record_id.isin(self.annot_db.record_id),'record_id']
        return remaining_rid.copy()
        
    def begin(self):
        print("You've got %d records to annotate" % (len(self.get_remaining_rid())))
        
        self.fig = plt.figure(figsize=(21,9))
        self.rec = self.get_next_record()
        if self.rec is None: return 
        self.wav = self.rec[self.channel]
        self.draw()
 
        'connect to all the events we need'
        self.cid_click = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect(
            'key_press_event', self.on_key)

        self.fig.canvas.manager.window.raise_()
        
    def stop(self):
        'disconnect all the stored connection ids'
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.fig.canvas.mpl_disconnect(self.cid_key)
        plt.close(self.fig)

    def draw(self):
        self.fig.clear()       
        ax=self.fig.add_subplot(111) 
        if self.plot_fun is not None:
            if isinstance(self.plot_fun,list):
                for i in range(len(self.plot_fun)):
                    self.plot_fun[i](self.rec)
            else:
                self.plot_fun(self.rec)
        else:
            ax.plot(self.wav.get_time(),self.wav)              
        ax.set_ylim(self.ylim)
        plt.tight_layout(pad=1, w_pad=2, h_pad=1)
        
    def on_click(self, event):
        if self.start is None:
            self.start=event.xdata
        else:
            print("start=%f end=%f" % (self.start,event.xdata))
            i=len(self.annot_current)
            self.annot_current.loc[i]=[int(self.rec.rid),i,self.start,event.xdata,self.comment]
            self.comment=''
            if self.wav is None:
                self.fig.axes[0].plot([self.start,event.xdata],[0,0],color="red")
            else:
                crop_wav=self.wav.crop(self.start,event.xdata)
                self.fig.axes[0].plot(crop_wav.get_time(),crop_wav,color="red")
            self.start=None

    def on_key(self,event):
        if event.key=='enter':
            #saving annotations to file            
            if len(self.annot_current) > 0:
                self.annot_db=self.annot_db.append(self.annot_current, ignore_index=True, verify_integrity=True)
                self.annot_current = pd.DataFrame(columns=self.col_names)            
            else:
                self.annot_current.loc[0]=[int(self.rec.rid),np.nan,np.nan,np.nan,self.comment]
                self.comment=''
                self.annot_db=self.annot_db.append(self.annot_current, ignore_index=True, verify_integrity=True)
                self.annot_current = pd.DataFrame(columns=self.col_names)            
            self.annot_db.to_csv(self.fname,sep='\t',index=False,na_rep='NA')

            #drawing new record
            self.rec = self.get_next_record()
            if self.rec is None: return 
            self.wav = self.rec[self.channel]
            self.draw()
            
        elif event.key=='backspace':
            self.annot_current = pd.DataFrame(columns=self.col_names) 
            self.draw()            
        elif event.key=='escape':
            self.stop()
        else:
            print(event.key)
            self.comment = self.comment + event.key
            

        

###### DEPRECATED METHODS #########################    


def detect(warble,channel='s',method='env',threshold=(0.04,0.01),env_method='jarne',preprocess=None,**kwargs):
    """
    detects regions of interest in the records of the warble object

    warble - Warble object to be annotated
    channel - string
        indicates the channel of the each record that will be used for the matching
    method - string
        Indicates the method used to find the regions of interest
        'env' calculates a smooth envelope and uses a threhold
    threshold - float or 2-tuple of floats
        upper/lower threshold for the segmentation
    env_method - string
        method to calculate de envelope
    preprocess - callable
        function defining pre-prcessing of the wav. Should take a Wav and return a Wav
        e.g. preprocess = lambda x: x.butter(low=300)
    kwargs - dict
        further arguments for envelope method. 
        By deault it will use method='jarne', bin_size=4415, cut_freq=5, padtype=None
        
    returns a pandas DataFrame with columns 'record_id', 'annot_id', 'start', 'end', 'comment'
    """
    warnings.warn("Annotate.detect is deprecated. Use Warble.detect instead")    
    
    col_names=['record_id','annot_id','start','end','comment']
    
    annot_db = pd.DataFrame(columns=col_names) 
    annot_db[['record_id','annot_id']] = annot_db[['record_id','annot_id']].astype(np.int64)
    
    if len(warble)==0:
        return annot_db
    
    #setting deault envelope method
    kwargs['method']=env_method
    if 'bin_size' not in kwargs: kwargs['bin_size']=4415
    if 'cut_freq' not in kwargs: kwargs['cut_freq']=5
    if 'padtype' not in kwargs: kwargs['padtype']=None
    
    if method=='env':
        #i=0
        #N=len(warble)
        #N_pb=int(math.ceil(N/100))

        for rec in progressbar(iter(warble)):
            #rec=warble[11]
            try:
                wav=rec[channel]
                if preprocess is not None:
                    wav=preprocess(wav)
                env=wav.envelope(**kwargs)
                annot_rec=detect_env(env,threshold=threshold)
                annot_rec.insert(0,'record_id',int(rec.rid))
                annot_db=annot_db.append(annot_rec, ignore_index=True, verify_integrity=True)
            except:
                warnings.warn("Record "+str(rec.rid)+" failed")
    else:
        raise Error('unknown method')

    col_names += [c for c in annot_db.columns.values if c not in col_names]
    annot_db=annot_db.reindex_axis(col_names,axis=1)
    annot_db.sort_values(['record_id','start'],inplace=True) 
    annot_db['annot_id']=annot_db.groupby('record_id')['annot_id'].transform(lambda x:np.arange(len(x)))
    annot_db.set_index(['record_id','annot_id'],drop=False,inplace=True,verify_integrity=True)
    return annot_db            
    
def match(warble,template,channel='s',method=None,preprocess=None,**kwargs):
    """
    [Deprecated] Match all records of a warble object to a template
    
    warble - Warble object or iterator over records
        collection of records to be matched
    template - Spe or Env object
        templated to be matched to 
    channel - string
        indicates the channel of the each record that will be used for the matching
    method - string
        Indicates the method used for the matching. Currently only 'spe' is implemented
        If None, it is inferred from the type of the temaplate object
    preprocess - callable
        function defining pre-prcessing of the wav. Should take a Wav and return a Wav
        e.g. preprocess = lambda x: x.butter(low=300)
    kwargs - dictionary
        further arguments for method
        
    returns a pandas DataFrame with columns 'record_id', 'annot_id', 'start', 'end', 'comment'
    """
    warnings.warn("Annotate.match is deprecated. Use Warble.match instead")    
    
    col_names=['record_id','annot_id','start','end','comment']
    annot_db = pd.DataFrame(columns=col_names) 
    
    if method is None:
        if isinstance(template,Spe): method='spe'
        elif isinstance(template,Env): method='env'

    if len(warble)==0:
        return annot_db

    #i=0
    #N=len(warble)
    #N_pb=int(math.ceil(N/100))
    #ToDo: this could be implemented more elegantly
    if method=='spe':
        for rec in progressbar(iter(warble)):
            #rec=warble[0]
            try:
                wav=rec[channel]
                if preprocess is not None:
                    wav=preprocess(wav)
                cdb = match_spe(wav.spectrogram(method=template.method,**template.method_args),template,**kwargs)
                cdb.insert(0,'record_id',rec.rid)
                annot_db=annot_db.append(cdb, ignore_index=True, verify_integrity=True)
            except:
                warnings.warn("Record "+str(rec.rid)+" failed")
        print('\ndone')

    elif method=='env':
        for rec in progressbar(iter(warble)):
            #rec=warble[0]
            try:
                wav=rec[channel]
                if preprocess is not None:
                    wav=preprocess(wav)
                cdb = match_env(wav.envelope(method=template.method,**template.method_args),template,**kwargs)
                cdb.insert(0,'record_id',rec.rid)
                annot_db=annot_db.append(cdb, ignore_index=True, verify_integrity=True)
            except:
                warnings.warn("Record "+str(rec.rid)+" failed")
        print('\ndone')    
    else:
        raise Error('unknown method')
        
    col_names += [c for c in annot_db.columns.values if c not in col_names]
    annot_db=annot_db.reindex_axis(col_names,axis=1)
    annot_db.sort_values(['record_id','start'],inplace=True) 
    annot_db['annot_id']=annot_db.groupby('record_id')['annot_id'].transform(lambda x:np.arange(len(x)))
    annot_db[['record_id','annot_id']] = annot_db[['record_id','annot_id']].astype(np.int64)
    return annot_db            
            

def match_spe(spe,template,threshold=None,threshold_factor=3,allow_overlap=False,max_ROI=100):
    """
    [Deprecated] Match a single spectrogram to a template
    
    spe - Spe object
        The spectrogram to be annotated
    template - Spe object
        The patter to matched to
    threshold - float
        The minimum correlation value to be assigned as a positive match
        If None it is set to basal+threshold_factor*std , where basal and std are 
        the median and a robust estimation of the standard deviation of the correlation values
    threshold_factor - float
        Factor that defines the correlation threshold. (Used if threshold is None.)
    allow_overlap - boolean or float
        indicates if overlaps of the template should be allowed
        if float in [0,1], it indicates the fraction of the motif length that is allowed to overlap
    max_ROI - int
        maximun number of regions of interest in a record
        
    returns a pandas DataFrame with columns 'annot_id', 'start', 'end','max_cor', 'comment'
    
    spectrogram(method='mlab',NFFT=512,noverlap=256)
    """
    warnings.warn("Annotate.match_spe is deprecated. Use Annotate.match_obj instead")    
    
    allow_overlap=float(allow_overlap)
    relative_threshold = threshold is None
    
    col_names=['start','end','max_cor','comment']
    annot_db = pd.DataFrame(columns=col_names) 
    cc=spe.correlate(template).set_delay(spe.delay)

    basal=np.median(cc.array) #robust estimaton of baseline
    if relative_threshold:
        std=4*np.median(np.abs(cc.array-basal)/0.6745) #robust estimater of dispersion
        threshold=basal+threshold_factor*std
    #if basal > threshold:
    #    warnings.warn('basal higher than threshold. Seeting basal=treshold')
    #    basal = threshold     
   
    search_match=True
    loop_count=0
    while search_match:
        loop_count+=1
        if loop_count > max_ROI:
            raise Error("More ROIs detected than max_ROI")

        cc_max=np.amax(cc.array)        
        cc_tmax=cc.time_max()
       
        if cc_max > threshold:
            overlap=0
            for i, row in annot_db.iterrows():
                if row.start <= cc_tmax+template.duration and row.end > cc_tmax:
                    overlap+=(min(cc_tmax+template.duration,row.end)-max(cc_tmax,row.start))/template.duration
            if overlap <= allow_overlap:
                annot_db.loc[len(annot_db)]=[cc_tmax,cc_tmax+template.duration,cc_max,'']          
                ccidx=cc._crop_idx(cc_tmax-template.duration/2,cc_tmax+template.duration/2)
                #cc[ccidx[0]:ccidx[1]+1]=basal      
                cc[ccidx[0]:ccidx[1]+1]=0      
     
            else:
                ccidx=cc._crop_idx(cc_tmax-template.duration/4,cc_tmax+template.duration/4)
                #cc[ccidx[0]:ccidx[1]+1]=basal            
                cc[ccidx[0]:ccidx[1]+1]=0
            #cc.plot()
        else:
            search_match = False

    annot_db.sort_values(by='start',ascending=True,inplace=True)
    annot_db.insert(0,'annot_id',np.arange(len(annot_db)))      
    
    return annot_db

#ToDo: match_env and match_spe could be a single function if Spe and Env both have the same interface    
def match_env(env,template,threshold=0.7,allow_overlap=False,max_ROI=100):
    """
    [Deprecated] Match a single envelope to a template
    
    env - Wav or Env object
        The envelope to be annotated
    template - Env (or Wav) object
        The patter to matched to
    threshold - float
        The minimum correlation value to be assigned as a positive match
    allow_overlap - boolean or float
        indicates if overlaps of the template should be allowed
        if float in [0,1], it indicates the fraction of the motif length that is allowed to overlap
    max_ROI - int
        maximun number of regions of interest in a record
        
    returns a pandas DataFrame with columns 'annot_id', 'start', 'end','max_cor', 'comment'
    """
    warnings.warn("Annotate.match_env is deprecated. Use Annotate.match_obj instead")    
    
    allow_overlap=float(allow_overlap)
    
    col_names=['start','end','max_cor','comment']
    annot_db = pd.DataFrame(columns=col_names) 
    cc=env.correlate(template).set_delay(env.delay)
   
    search_match=True
    loop_count=0
    while search_match:
        loop_count+=1
        if loop_count > max_ROI:
            raise Error("More ROIs detected than max_ROI")

        cc_max=np.amax(cc.array)        
        cc_tmax=cc.time_max()
       
        if cc_max > threshold:
            overlap=0
            for i, row in annot_db.iterrows():
                if row.start <= cc_tmax+template.duration and row.end > cc_tmax:
                    overlap+=(min(cc_tmax+template.duration,row.end)-max(cc_tmax,row.start))/template.duration
            if overlap <= allow_overlap:
                annot_db.loc[len(annot_db)]=[cc_tmax,cc_tmax+template.duration,cc_max,'']          
                ccidx=cc._crop_idx(cc_tmax-template.duration/2,cc_tmax+template.duration/2)
                #cc[ccidx[0]:ccidx[1]+1]=basal      
                cc[ccidx[0]:ccidx[1]+1]=0      
     
            else:
                ccidx=cc._crop_idx(cc_tmax-template.duration/4,cc_tmax+template.duration/4)
                #cc[ccidx[0]:ccidx[1]+1]=basal            
                cc[ccidx[0]:ccidx[1]+1]=0
            #cc.plot()
        else:
            search_match = False

    annot_db.sort_values(by='start',ascending=True,inplace=True)
    annot_db.insert(0,'annot_id',np.arange(len(annot_db)))      
    
    return annot_db

def progressbar(iter_):
    """
    Simple progress bar utility
    """
    i=1; N_pts=0
    N=len(iter_)
    N_pb=int(math.ceil(N/100))
    msg="\n| 0% - Annotating "+str(N)+" records -"
    print(msg+" "*(93-len(msg))+"100% |",end="\n",flush=True)
    yield next(iter_)
    while True:
        i+=1
        if i%N_pb==0:
            N_new=int(math.floor(100*i/N)-N_pts)
            print('\b',"."*N_new,sep="",end=" ",flush=True)
            N_pts+=N_new
        if N_pts>=100:
            print(end="\n",flush=True)
        yield next(iter_)
        
