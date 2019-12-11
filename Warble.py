#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WarblePy package was developed in the Laboratory of Dynamical Systems, University
of Buenos Aires, Argentina by Alan Bush. The package is loosley inspired in 
warbleR by Marcelo Araya-Sala (hence the name).  

The package provides functionality for the analysis of EMG data from sleeping 
and singing birds. 

This module defines the Warble object, which represents a collection of records 
for a given bird. 

@author: Alan Bush
"""
import os
import fnmatch
import pandas
import numpy
import re
import collections
import copy
import math
import scipy.io.wavfile
import warnings
import Annotate
import pdb

from Wav import Wav, WavPromise, Warped
from Rec import Rec, ann_iter
from Spe import Ens
            

class Error(Exception): pass

class Warble(collections.abc.Sequence):
    def __init__(self,record_db,file_db,lazy_load=False,preprocess=None,**kwargs):
        self.record_db=record_db
        self.file_db=file_db
        self.annot=dict()
        if preprocess is None:
            self.preprocess=dict()
        else:
            self.preprocess=preprocess
        self.lazy_load=lazy_load
        self.__dict__.update(kwargs)
        self.is_invisible=False

    @classmethod
    def load(cls, paths, channels=['s'], log_files="AVIMAT-log.txt", wav_files="*.wav",
             date_format="%Y-%m-%d %H.%M.%S", rename_log_cols=None, calculate_duration=False, 
             log_required=False, allow_unknown_channels=True, lazy_load=False, preprocess=None):
        """
        load a warble object
        
        paths: (list of) string(s) containing the paths to the mother folder(s) 
                that contains the child folders with the wav files. 
        channels: list of strings specifying the names of the channels to be loaded
                the log file is expected to have columns named ch_X_file where X is the 
                channel specifier. Use rename_log_cols to satisfy this requirement. 
        log_files: name or pattern expected for the log files inside the child folders
        wav_files: glob pattern for thewav files
        date_format: format expected for the datetime, after pasting the 'date' and 'time' 
                     columns of the log file
        rename_log_cols: dictionary with eich to rename the log columns as to math the expected
                         format. Should be of the form {'old_name':'new_name'}            
        calculate_duration: boolean indicating if individual wav files should be loaded to calculate
                            their duration in seconds
        log_required: boolean indicating if a log file is required in order to load the wav 
                      files inside a child folder
        allow_unknown_channels: boolean indicating if files of unknown channel (not related to any log file)
                      should be allowed or removed. 
        lazy_load: bool indicating if the wavs should be loaded in lazy fashion or in an eager manner
                    sets the default behaviour for get_record
        preprocess: dict of function to preprocess each channel. The key should corresponde to the channel's name
                    each element of the dict should be a function that accepts a Wav and returns a Wav
        """
        #paths=['/media/alan/Storage/Data-LSD/2016-06-30-zfAB005-vi/wavs/','/media/alan/Storage/Data-LSD/2016-06-30-zfAB005-vi/playback-templates/']
        
        #entry=next(os.scandir(path))
        #loading logs and wavs to pandas data frames
        log_df_list=[]
        wav_df_list=[]
        log_path_list=[]
        wav_path_list=[]
        print("Loading folders:")
        for path in paths:
            #path = paths[0]
            if os.path.exists(path):
                for entry in os.scandir(path):
                    #[entry.name for entry in os.scandir(path)]
                    has_log=False
                    if entry.is_dir():
                        print(path+entry.name)
                        logs=fnmatch.filter(os.listdir(entry.path),log_files)
                        wavs=fnmatch.filter(os.listdir(entry.path),wav_files)
                        for log in logs:
                            log_path_list.append(entry.name)
                            log_df=pandas.read_table(os.path.join(entry.path,log))        
                            log_df_list.append(log_df)
                            has_log=True
                        if (not log_required or has_log) and (len(wavs)>1):  
                            wav_path_list.append(entry.name)
                            wav_df=pandas.DataFrame({'path':path,'folder':entry.name,'file':wavs})
                            wav_df_list.append(wav_df)

        #sorting alphabetically
        log_df_list=[x for _,x in sorted(zip(log_path_list,log_df_list))]
        wav_df_list=[x for _,x in sorted(zip(wav_path_list,wav_df_list))]
        
        #consolidating record_db and file_db        
        record_db=pandas.concat(log_df_list,ignore_index=True)    
        #record_db.sort_values(['date','time'],inplace=True)
        record_db=record_db.reset_index(drop=True)
        if rename_log_cols is not None:
            record_db.rename(columns=rename_log_cols,inplace=True)
        record_db['oldrid']=record_db.index #old record_id assignation. Not reliable
        
        date_nstr=record_db.date.str.replace("[^0-9]","",-1)
        time_nstr=record_db.time.str.replace("[^0-9]","",-1)
        record_db.insert(0,'record_id',date_nstr.str.cat(time_nstr).astype(numpy.int64))
        record_db.sort_values('record_id',inplace=True)        
        record_db.set_index('record_id',drop=False,inplace=True,verify_integrity=True)        
        record_db.index.rename('record_idx',inplace=True)
        
        file_db=pandas.concat(wav_df_list,ignore_index=True) 
        
        dup_file=file_db.duplicated(subset='file')
        if any(dup_file):
            print("\nDuplicated file names:")
            print(file_db.loc[dup_file,'file'])
            print("eliminating duplicated entries")
            file_db.drop_duplicates(subset='file',inplace=True)
        file_db=file_db.reset_index(drop=True)
        file_db.insert(0,'file_id',file_db.index)

        #calculating duration
        if calculate_duration:
            print("\nCalculating duration of each wav... ")
            def get_duration(reg):
                fs, input_array = scipy.io.wavfile.read(os.path.join(reg.path,reg.folder,reg.file))
                return input_array.shape[0]/fs            
           
            file_db['duration']=file_db.apply(get_duration,axis=1)
            print("done")

        #parsing date and time
        assert 'date' in record_db.columns
        assert 'time' in record_db.columns
        record_db["datetime"]=pandas.to_datetime(record_db.date + ' ' + record_db.time,format=date_format)
        
        file_data_list=[]
        for ch in channels:
            #i_ch=iter(channels)
            #ch = next(i_ch)
            ch_file_db=file_db[['file_id','file']].rename(columns={'file_id':'ch_'+ch+'_id','file':'ch_'+ch+'_file'})                       
            record_db=record_db.merge(ch_file_db,on='ch_'+ch+'_file',how='left',copy=False)
            ch_vars=set(record_db.columns).intersection(set(['ch_'+ch+'_'+var for var in ['min','max']]))
            file_data=record_db.loc[pandas.notnull(record_db['ch_'+ch+'_id']),['ch_'+ch+'_id']+list(ch_vars)]
            #setting defaults
            if not 'ch_'+ch+'_min' in ch_vars: file_data['ch_'+ch+'_min']=numpy.nan
            if not 'ch_'+ch+'_max' in ch_vars: file_data['ch_'+ch+'_max']=numpy.nan
            file_data['channel']=ch
            file_data.rename(columns={'ch_'+ch+'_id':'file_id','ch_'+ch+'_min':'min','ch_'+ch+'_max':'max'},inplace=True)
            file_data=file_data[['file_id','channel','min','max']]
            file_data_list.append(file_data)
            #deleting columns copied to file_db from record_db
            del record_db['ch_'+ch+'_file']
            for var in ch_vars: del record_db[var]

        file_data=pandas.concat(file_data_list,ignore_index=True)
        file_data['file_id']=file_data['file_id'].astype(numpy.int32)
        file_data.drop_duplicates(subset='file_id',inplace=True)
        
        file_data.set_index('file_id',drop=True,inplace=True,verify_integrity=True)
        file_db.set_index('file_id',drop=False,inplace=True,verify_integrity=True)
        file_db=file_db.join(file_data)
        file_db.index.rename('file_idx',inplace=True)

        file_db_na_channel=file_db.loc[file_db.channel.isnull(),('file','folder')]
        if len(file_db_na_channel) > 0 and not allow_unknown_channels:
            print("\nRemoving registers with unknown channel from files_db")
            print(file_db_na_channel)
            file_db.dropna(axis=0,subset=['channel'],inplace=True) 

        record_db.sort_values('record_id',axis=0,inplace=True)                
        record_db.set_index('record_id',drop=False,inplace=True,verify_integrity=True)
        record_db.index.rename('record_idx',inplace=True)
        return cls(record_db,file_db,lazy_load=lazy_load,preprocess=preprocess)
        
            
    def add_file_info_to_record_db(self, channel, column):
        """
        [Deprecated] Add file information to the record_db
        
        channel - str
            indicates the channel of interest (subset of he file_db)
        column - str
            indicates the column of interest of the file_db
        
        returns the modified warble object
        """
        warnings.warn("Deprecated function. Use Warble.add_file_info instead")
        file_info=self.file_db.loc[self.file_db.channel==channel,("file_id",column)]
        file_info.rename(columns={'file':'ch_'+channel+'_'+column,'file_id':'ch_'+channel+'_id'},inplace=True)
        file_info.set_index('ch_'+channel+'_id',inplace=True,drop=True)
        self.record_db=pandas.merge(self.record_db,file_info,how='left',on='ch_'+channel+'_id',right_index=True)
        return self.invisible()

    def add_file_info(self, to='record_db', channel='s', column='file', prefix=True):
        """
        Add file information to the record_db
        
        channel - str
            indicates the channel of interest (subset of he file_db)
        column - str
            indicates the column of interest of the file_db
        prefix - bool
            indicates if 'channel' prefix should be added to column name
        
        returns the modified warble object
        """
        new=column
        if prefix is True:
            new='ch_'+channel+'_'+column
            
        file_info=self.file_db.loc[self.file_db.channel==channel,("file_id",column)]
        file_info.rename(columns={column:new,'file_id':'ch_'+channel+'_id'},inplace=True)
        file_info.set_index('ch_'+channel+'_id',inplace=True,drop=True)
     
        new_in_record_db=new in self.record_db.columns.values
        if new_in_record_db:
            del self.record_db[new]
        
        self.record_db=pandas.merge(self.record_db,file_info,how='left',on='ch_'+channel+'_id',right_index=True)

        if to in self.annot:
            if new in self.annot[to].columns.values:
                del self.annot[to][new]
            self.annot[to]=self.annot[to].join(self.record_db[new])
            if not new_in_record_db:
                del self.record_db[new]
        elif not to == 'record_db':
            raise Error("'to' should be either 'record_db' or a member of self.annot")
            
        return self.invisible()


    def register_channel(self, mapping_db, copy_delays=True):
        """
        Registers a channel to the datasets
        
        mapping_db - pandas data frame
            first column is expected to contain the filenames of the channel already
            present in record_db. The second column is expected to have the filenames 
            of the new channel to be added. Note that the these files have to be present
            in file_db, normally with unknown channel. Make sure to use 
            allow_unknown_channels=True when loading the Warble object.
            The columns names should have de form ch_*_file, with * defining the channel 
            name.
            
        copy_delays - boolean
            Indicates if the delay times of the first channel should be copied to the newly
            registered channel
        """
        assert mapping_db.shape[1]==2
        cols = mapping_db.columns.values.tolist()
        matchs = [re.match('ch_(?P<channel>\w+)_file',col) for col in cols] 
        for col,m in zip(cols,matchs):
            if m is None:
                raise Error(col + " doesn't conform to the ch_*_file format")
        ch1, ch2 = (m.group('channel') for m in matchs)
        if ch1 not in self.channels:
            raise Error('channel ' + ch1 + 'not present in Warble object')
        
        #registering files of the new channel
        for nf in mapping_db[cols[1]].dropna().unique():
            #nf=mapping_db[cols[1]].dropna().unique()[0]
            f_db=self.file_db.loc[self.file_db.file==nf]
            if len(f_db) == 0:
                raise Error("attempting to register file "+nf+" not present in the Warble object")
            self.add_file(f_db.path.iloc[0],f_db.folder.iloc[0],f_db.file.iloc[0],channel=ch2)

        mdb=self.ch_file2id(mapping_db)
        if any(mdb.groupby('ch_'+ch1+'_id').apply(lambda df: len(df['ch_'+ch1+'_id'].unique()))>1):
            raise Error("ambiguous mapping_db")
        mdb=mdb.groupby('ch_'+ch1+'_id').apply(lambda df: df.iloc[0])
        mdb.set_index('ch_'+ch1+'_id',inplace=True,verify_integrity=True,drop=True)
        
        if 'ch_'+ch2+'_id' in self.record_db.columns:
            del self.record_db['ch_'+ch2+'_id']
        self.record_db = self.record_db.join(mdb,on='ch_'+ch1+'_id',how='left')
        if copy_delays: 
            self.record_db['ch_'+ch2+'_delay']=self.record_db['ch_'+ch1+'_delay']
        self.record_db.set_index('record_id',drop=False,inplace=True,verify_integrity=True)
        self.record_db.index.rename('record_idx',inplace=True)
        return self.invisible()
        
    def add_file(self, path, folder, file, channel, log="_warble-normalization.dat"):
        """
        Adds a file to file_db
        
        path: string with path to parent folder
        folder: string child folder name
        file: string file name of the wav to add
        channel: string channel name of the added channel
        log: string, filename of the log file as saved by Wav.write
        """
        
        min_=numpy.nan
        max_=numpy.nan
        
        if log is not None: 
            log_fname = os.path.join(path,folder,log)
            if os.path.isfile(log_fname):
                log_db=pandas.read_table(log_fname)
                if any(log_db.file==file):
                    max_=log_db.loc[log_db.file==file].norm_max.iloc[0]
                    min_=log_db.loc[log_db.file==file].norm_min.iloc[0]
                
        fname=os.path.join(path,folder,file)
        if not os.path.isfile(fname): raise Error("file not found")
        fdict={'file':file,'folder':folder,'path':path, \
               'channel':channel,'min':min_,'max':max_}
        
        if any(self.file_db.columns=='duration'):
            fs, input_array = scipy.io.wavfile.read(fname)
            fdict['duration'] = input_array.shape[0]/fs 
        
        #cheking if file is already in file_db (assuming unique file names)
        n_match=(self.file_db.file==file).sum()
        if n_match==0: #new entry
            fdict['file_id']=self.file_db.file_id.max()+1
            self.file_db=self.file_db.append(fdict,ignore_index=True)          
        elif n_match==1: #existing entry
            #idx=self.file_db.loc[self.file_db.file==file].index
            for col in fdict:
                #self.file_db=self.file_db.set_value(idx,col,fdict[col])
                self.file_db.at[self.file_db.file==file,col] = fdict[col]
        else:
            raise Error(file + " found " + str(n_match) + "on file_db. Filenames should be unique." )


    def ch_file2id(self,df,inplace=False):
        """
        Replaces file names by file ids according to file_db
        
        df - pandas data frame to be replaced. Columnes should be names as ch_*_file.
        inplace - boolean indicating if the replacement should be done inplace
        """
        if not inplace:
            df=df.copy()
        
        col_names=df.columns.values
        for col in col_names:
            mo=re.match('ch_(.*)_file',col)
            if mo is not None:
                ch=mo.group(1)
                df=df.merge(self.file_db.loc[self.file_db.channel==ch,('file_id','file')],left_on=col,right_on="file",how='left')
                df.drop('file', axis=1, inplace=True)
                df.rename(columns={'file_id':'ch_'+ch+'_id'},inplace=True)  
                df.drop(col, axis=1, inplace=True)
        
        return df
        
    def subset(self, select=None, by='record_db', inplace=False, **kwargs):
        """
        Subset the Warble object. Returns a smaller Warble object with selected records.
        
        select - string or bool series 
            If it is a string it should evaluate to a boolean in the conntext of the 'by' DataFrame
            If it is a series it should be of length appropiate for 'by' DataFrame
            If None all registers in the 'by' DataFrame will be selected
        by - string
            Indicates the DataFrame by which to subset
        inplace - bool
            If True the subsetting is done on the object, if False returns a copy of the object subseted
        **kwargs
            Further arguments for pandas.DataFrame.query
        """

        #dealing with inplace
        if inplace:
            output=self
        else:
            output=copy.deepcopy(self)
        #selecting DataFrame by which to filter
        if by in ['record_db','file_db']:
            by_db=getattr(output,by).copy()
        elif by in self.annot:
            by_db=self.annot[by].copy()
        else:
            raise Error("unknown 'by' DataFrame to filter")
        #filtering
        if select is not None:
            if type(select)==str: 
                by_db.query(select,inplace=True, **kwargs)
            else:
                by_db=by_db.loc[select]
                      
        #filtering record_db and annot DataFrames
        criteria = output.record_db.record_id.isin(by_db.record_id)
        output.record_db=output.record_db.loc[criteria]
        
        for annot_type in output.annot:
            output.annot[annot_type]=output.annot[annot_type].loc[output.annot[annot_type].record_id.isin(by_db.record_id)]

        if by in output.annot:
            output.annot[by]=by_db    
                         
        return output

    def sample(self,n=1,**kwargs):
        """
        Returns a sample of the Warble object with 'n' records
        n - integer
            number of random samples to draw
        kwargs
            further arguments for DataFrame.sample
        """
        output=copy.deepcopy(self)
        output.record_db=output.record_db.sample(n=n,**kwargs)
        for annot_type in output.annot:
            output.annot[annot_type]=output.annot[annot_type].loc[output.annot[annot_type].record_id.isin(output.record_db.record_id)]
        return(output)
        
    def get_record(self, record_id, lazy_load=None, preprocess=True):  
        """
        Retrieves a record by 'record_id'
        """
        if preprocess is True: preprocess=self.preprocess
        if lazy_load is None: lazy_load=self.lazy_load
        rec_record=self.record_db.loc[record_id,:].to_dict()
        rec_channels={}
        for key in rec_record:
            m = re.match("^ch_(.*)_id$",key)
            if m and pandas.notnull(rec_record[key]):
                ch=m.group(1)
                ch_delay="ch_"+ch+"_delay"
                fd=self.file_db.loc[int(rec_record[key])]
                delay=0
                if ch_delay in rec_record: delay=rec_record[ch_delay]
                if lazy_load:
                    rec_channels[ch]=WavPromise(os.path.join(fd['path'],fd['folder']),fd['file'],\
                                          norm_max=fd['max'],norm_min=fd['min'],delay=delay,channel=ch,\
                                          norm=bool(pandas.notnull(fd['max']) and pandas.notnull(fd['min'])))
                    if ch in preprocess:
                        rec_channels[ch].preprocess=preprocess[ch]
                else:
                    rec_channels[ch]=Wav.read(os.path.join(fd['path'],fd['folder'],fd['file']),\
                                          norm_max=fd['max'],norm_min=fd['min'],delay=delay,channel=ch,\
                                          norm=bool(pandas.notnull(fd['max']) and pandas.notnull(fd['min'])),\
                                          log=None)
                    if ch in preprocess:
                        rec_channels[ch]=preprocess[ch](rec_channels[ch])
                        
        annot_dict={annot_type:self.annot[annot_type].loc[record_id] for annot_type in self.annot if record_id in self.annot[annot_type].index}
        return Rec(record=rec_record,annot=annot_dict,**rec_channels)

    #dealing with annotations        
    def load_annot(self,*args,**kwargs):
        warnings.warn("deprecated, use annot_load instead")
        return self.annot_load(*args,**kwargs)
        
    def annot_load(self, annot="all", path="./annotations", extension=".dat"):
        """
        Loads and registers to the Warble obect the specified annotation files
        
        annot - str or list of str
            annotation files to load. If 'all', all files in the path are loaded
        path - str
            path to the annotations folder
        extension - str
            extension for the annotation files
        
        return a warble object with registered annotations    
        """
        if annot=='all':
            annot_files=fnmatch.filter(os.listdir(path),"*"+str(extension))
        else:
            if not isinstance(annot,list):
                annot=[annot]
            annot_files=[f+extension for f in annot if os.path.isfile(os.path.join(path,f+extension))]

        for annot_file in annot_files:    
            self.annot_register(annot_file[0:-len(extension)],pandas.read_table(os.path.join(path,annot_file)))
        return self.invisible()

    def write_annot(self,*args,**kwargs):
        warnings.warn("deprecated, use annot_write instead")
        return self.annot_write(*args,**kwargs)        

    def annot_write(self, annot='all', path="./annotations", extension=".dat"):
        """
        Writes specified annotation DataFrames
        
        annot - str or list of str
            annot DataFrames to be writen. The special keyword 'all' writes all annots. 
        path - str
            path to the annotations folder
        extension - str
            extension for the annotation files
        
        returns the warble object
        """
        if annot=='all':
            annot=list(self.annot.keys())
        if not isinstance(annot,list):
            annot=[annot]
        if not os.path.exists(path):
            os.makedirs(path)
        for fn in annot:
            self.annot[fn].to_csv(os.path.join(path,str(fn)+str(extension)),sep='\t',index=False,na_rep='NA')
        return self.invisible()
        

    def register_annot(self,*args,**kwargs):
        warnings.warn("deprecated, use annot_register instead")
        return self.annot_register(*args,**kwargs)   
        
    def register_annot_db(self,annot_type,annot_db):
        warnings.warn("deprecated, use annot_register instead")
        return self.annot_register(annot_type,annot_db)
            
    def annot_register(self,annot_type,annot_db):
        """
        Registers a annotation DataFrame to the warble object
        
        annot_type - str
            the key name for the annotation data frame
        annot_db - pandas.DataFrame
            table with the annotations. Should contain columns 'record_id','annot_id','start','end'
        """
        if annot_db.shape[0] == 0:
            warnings.warn("Nothing to register")
            return self
        col_names=['record_id','annot_id','start','end']
        if not isinstance(annot_db,pandas.DataFrame):
            raise Error("annot_db should be instance of pandas.DataFrame")
        if not set(col_names).issubset(annot_db.columns):
            raise Error("columns 'record_id','annot_id','start','end' should be present in annot_db")
        if not 'comment' in annot_db.columns:
            annot_db['comment']=""
        annot_db=annot_db.dropna(axis=0,subset=['annot_id']).copy()
        annot_db[['record_id','annot_id']] = annot_db[['record_id','annot_id']].astype(numpy.int64)
        annot_db.sort_values(['record_id','annot_id'],inplace=True) 
        annot_db['comment'] = annot_db['comment'].astype(str)
        annot_db.set_index(['record_id','annot_id'],drop=False,inplace=True,verify_integrity=True)
        annot_db.index.rename(['record_idx','annot_idx'],inplace=True)
        annot_db['duration']=annot_db['end']-annot_db['start']
        if 'datetime' in annot_db.columns:
            del annot_db['datetime']
        annot_db=annot_db.join(self.record_db.datetime,on='record_id',how='left')
        annot_db['datetime']+=pandas.to_timedelta(annot_db['start'],unit='s')
        col_names += ['duration','datetime','comment']
        col_names += [c for c in annot_db.columns.values if c not in col_names]
        #annot_db=annot_db.reindex_axis(col_names,axis=1)
        annot_db=annot_db.reindex(col_names,axis=1)
        annot_db.name=annot_type
        self.annot[annot_type]=annot_db
        return self.invisible()        
        
    def get_annot(self,*args,**kwargs):
        return self.annot_get(*args,**kwargs)   
               
    def annot_get(self, annot=None, annot_id=None, record_id=None, annot_type=None,\
                  select=None, start=None, end=None, lazy_load=None, fetch=None, flanking=0):
        """
        Retrieves an annotation
        
        annot - str or pandas.Series
            if pandas.Series, it should contain elements 'record_id', 'annot_id', 'start' and 'end' 
            if str: name of the annotation database in the warble.annot dictionary
            if 'annot' and 'annot_id' are None, 'start' and/or 'end' are required
        record_id - int
            id of the record (YYYYMMDDhhmmss)
        start - float
            start point in seconds of the annotation
            if 'start' and 'end' are None, 'annot' and 'annot_id' are required
        end - float
            end point in seconds of the annotation
            if 'start' and 'end' are None, 'annot' and 'annot_id' are required
        annot_id - int
            id of the annotation within the record
            if 'annot' and 'annot_id' are None, 'start' and/or 'end' are required
            if annot_id='any', the first annotation matching the criteria is returned
        select - str of pandas.Series of booleans
            selection criteria of a single row of the annot DataFrame. 
            Used if 'annot' is a str and 'annot_id' is None
        lazy_load - bool
            should lazy loadign be used?
        annot_type - str
            type of annotation. If None and annot is a str, annot_type is set to annot
        fetch - str or None
            Indicates how to deal with ambiguous specification of annotations. 
            If None a warwing is issue if multiple annotations match the selection
            and the first one is fetched. 
            If 'first', the first annotations is fetched without warning.
            if 'random', a random annotation is fetched, without a warning
            If 'unique', a error is raised if more than one annotations matched the selection
        flanking - float or 2-tuple of floats
            time in seconds by which to flank the annotation
            if 2-tuple, the first element extends the annot by this amount at the begging
            and the second element extends the annot at the end. Negative values reduce
            the annot.
            
        returns a Ann object            
        """
        if fetch is not None:
            warnings.warn("'fetch' argument not implemented yet. Use annot_id='any'")
        
        if annot is not None:
            if isinstance(annot, pandas.Series):
                record_id=annot.record_id
                annot_id=annot.annot_id
                start=annot.start
                end=annot.end
            elif isinstance(annot,str):
                if not annot in self.annot:
                    raise Error("annot not found in self.annot")

                if select is not None:
                    if isinstance(select,str):
                        sel_db=self.annot[annot].copy().query(select)        
                    elif isinstance(select,pandas.Series):
                        sel_db=self.annot[annot].copy().loc[select,:]        
                    else:
                        raise Error("Invalid type for select")
                else:
                    sel_db=self.annot[annot].copy()

                if record_id is not None:
                    sel_db=sel_db.query("record_id=="+str(record_id))
                    
                if len(sel_db)==0:
                    raise Error("No annotations match your selection")
                    
                if annot_id is None or annot_id=='any':
                    if len(sel_db)>1 and annot_id is None:
                        warnings.warn(str(len(sel_db)) + " annotations match your selection. Using first.")

                    annot_id=sel_db.iloc[0].annot_id
                    record_id=sel_db.iloc[0].record_id

                if record_id is None:
                    #checking if record_id is well determined by annot_id
                    sel_annot_db=sel_db.loc[sel_db.annot_id==annot_id,:]
                    if len(sel_annot_db)==1:
                        record_id=sel_annot_db.iloc[0].record_id   
                    else:
                        raise Error("if annot is a str and annot_id is ambiguous, record_id should be given")
                if not (record_id,annot_id) in sel_db.index:
                    raise Error("selected 'record_id' and 'annot_id' not in annotation db")
                if annot_type is None:
                    annot_type=annot
                annot=sel_db.loc[record_id,annot_id]
        else:
            if record_id is None:
                raise Error("if annot is None, record_id should be given")            
            if start is None and end is None:
                raise Error("if annot is None, either start or end should be given")            
            if annot_id is None:
                annot_id=0
            annot = pandas.Series({'record_id':record_id,'annot_id':annot_id,'start':start,'end':end})
        rec = self.get_record(record_id,lazy_load)
        return rec.get_annot(annot,annot_type=annot_type,flanking=flanking)
     
    def calculate(self, db='record_db', select=None, flanking=0, augment=False, values=None, **kwargs):
        """
        Calculates scalar values from records or annotations
        
        db - str
            name of the DataFrame on which the calculation will be done
            may be either 'record_db' or a member of self.annot
        select - str of pandas.Series of boolen 
            selection of db
        flanking - float
            time in seconds by which to flank the annotation (used if db is in self.annot)
        augment - bool
            indicates if the calculation should take on from the las record done by previous calculations
            If True, it only attemps to do the calculation for rows with NaN at the specified column.
            (NaN previous to teh last valid value are ignored. i.e. it only calculates at the end of the DataFrame)
            If False all rows are (re)calculated.
        values - callable
            function used to do several calculations. The functions should accept a Rec or Ann and return a dictionary.
            The keys ares used as the dataframe columns
        kwargs - dict
            functions responsible for the calculation. The functions should accept a Rec or Ann and return a scalar value.
            The name of the entry is used as the new column name in the dataframe
            e.g.: s_abs_sum=lambda x: np.sum(np.abs(x.s))
        
        returns the reference to the modified warble object
        """
        if db=='record_db':
            for var in kwargs:
                last_rid=0
                if var not in self.record_db.columns:
                    self.record_db[var]=numpy.nan
                elif augment is True:
                    last_rid=self.record_db[var].notnull()[::-1].idxmax()
                    
                for rec in progressbar(iter(self.subset(select,by='record_db').subset("record_id>"+str(last_rid),by='record_db'))):
                    try:
                        _=self.record_db.set_value(rec.rid,var,kwargs[var](rec));
                    except Exception as err:
                        _=self.record_db.set_value(rec.rid,var,numpy.nan);
                        warnings.warn("Error calculating rid "+str(rec.rid)+"\n"+str(err))
                if values is not None:
                    raise Error("'Values' argument not implemented for 'record_db' yet")
        else:
            if db not in self.annot:
                raise Error("db should be either 'record_db' or a member of self.annot")
            for var in kwargs:
                last_rid=0
                if var not in self.annot[db].columns:
                    self.annot[db][var]=numpy.nan
                elif augment is True:
                    #FIXME: I think this doesn't work as expected
                    last_rid=self.annot[db][var].notnull()[::-1].idxmax()[0]

                for ann in progressbar(self.iter_annot(db,select,_select="record_id>"+str(last_rid),flanking=flanking)):
                    try:
                        _=self.annot[db].set_value((ann.rid,ann.aid),var,kwargs[var](ann));
                    except Exception as err:
                        _=self.annot[db].set_value((ann.rid,ann.aid),var,numpy.nan);
                        warnings.warn("Error calculating rid "+str(ann.rid)+ ", aid"+str(ann.aid)+"\n"+str(err))
                        
            if values is not None:
                #initializing variables
                i1=self.iter_annot(db,select,flanking=flanking)
                eg=None
                last_rid=0     
                try_count=0
                
                while eg is None and try_count<10: 
                    try: 
                        eg=values(next(i1))
                    except:
                        eg=None
                        try_count+=1

                for var in eg:
                    if isinstance(eg[var],str):
                        eg[var]=''
                    else:                        
                        eg[var]=numpy.nan

                if not set(self.annot[db].columns).issubset(set(eg.keys())):
                    #some or all variables not present. Recalculating all.
                    for var in eg:
                        self.annot[db][var]=eg[var]
                elif augment is True:
                    #FIXME: I think this doesn't work as expected
                    last_rid=self.annot[db][list(eg.keys())[0]].notnull()[::-1].idxmax()[0]        

                for ann in progressbar(self.iter_annot(db,select,_select="record_id>"+str(last_rid),flanking=flanking)):
                    try:
                        v=values(ann)
                    except Exception as err:
                        v=eg
                        warnings.warn("Error calculating rid "+str(ann.rid)+ ", aid"+str(ann.aid)+"\n"+str(err))            
                    for var in v:
                        _=self.annot[db].set_value((ann.rid,ann.aid),var,v[var]); 

        return self.invisible()

    def transform(self, db='record_db', by=None, **kwargs):
        """
        Transforms columns of DataFrames in the Warble object
        
        db - str
            name of the DataFrame on which the calculation will be done
            may be either 'record_db' or a member of self.annot
        by - str or list of str
            by operator, not implemented yet
        kwargs - dict
            arguments define the new columns to be created. The value way be
            a str to be evaluated in the context of the DataFrame
            a pandas.Series to placed as is in the DataFrame
            a callable to be applied on the DataFrame
            e.g.: duration='end-start' or duration=lambda row: row.end-row.start
        
        returns the reference to the modified warble object
        """
        if by is not None:
            raise Error("transform 'by' not implemented yet")
        else:
            if db=='record_db':    
                for var in kwargs:
                    if isinstance(kwargs[var],str):
                        self.record_db[var]=self.record_db.eval(kwargs[var])
                    elif isinstance(kwargs[var],pandas.Series):
                        self.record_db[var]=kwargs[var]
                    elif callable(kwargs[var]):
                        self.record_db[var]=self.record_db.apply(kwargs[var],axis=1)
                    else:
                        raise Error("unknown type for argument "+var+". Should be a str, pandas.Series or a callable.")        
            elif db in self.annot:
                for var in kwargs:
                    if isinstance(kwargs[var],str):
                        self.annot[db][var]=self.annot[db].eval(kwargs[var])
                    elif isinstance(kwargs[var],pandas.Series):
                        self.annot[db][var]=kwargs[var]
                    elif callable(kwargs[var]):
                        self.annot[db][var]=self.annot[db].apply(kwargs[var],axis=1)
                    else:
                        raise Error("unknown type for argument "+var+". Should be a str, pandas.Series or a callable.")                 
            else:
                raise Error("db should be either 'record_db' or a member of self.annot")
        return self.invisible()
 
    def iter_annot(self,*args,**kwargs):
        return self.annot_iter(*args,**kwargs)   
        
    def annot_iter(self,annot_type,select=None,_select=None,flanking=0, **kwargs):
        """
        Creates an iterator over annotations
        
        annot_type - str
            name of the annotation dataframe in warble.annot
        select - str or boolean pandas.Series
            filter over the dataframe before the iterator is created
        _select - str
            filter over the dataframe to be applied after 'select'
        flanking - float or 2-tuple of floats
            time in seconds by which to flank the annotation
            if 2-tuple, the first element extends the annot by this amount at the begging
            and the second element extends the annot at the end. Negative values reduce
            the annot.
        kwargs - dict
            further arguments for pandas.DataFrame.query if select is a str
            
        returns an iterator over annotations
        """
        if not annot_type in self.annot:
            raise Error(str(annot_type)+" not in warble.annot")
        db=self.annot[annot_type].copy()    
        #filtering
        if select is not None and len(db)>0:
            if type(select)==str: 
                db.query(select,inplace=True, **kwargs)
            else:
                db=db.loc[select,:]
        if _select is not None and len(db)>0:
            db.query(_select,inplace=True, **kwargs)
        return ann_iter_warble(self,db,annot_type,flanking)
        
        
    def _init_annot(self,db,annotate,select,augment,flanking):
        """
        initializes the annotation process
  
        see parameter definition of Warble.detect or Warble.match
        
        returns a tupple containing the rec/ann iterator, empty annot_db and col_names
        """
     
        col_names=['record_id','annot_id','start','end','comment']
        annot_db=pandas.DataFrame(columns=col_names) 
        last_rid=0

        if annotate is not None:
            if annotate in self.annot and augment is True:
                annot_db = self.annot[annotate]
                last_rid=annot_db.iloc[-1].record_id
                if 'datetime' in annot_db.columns:
                    del annot_db['datetime']
                if 'duration' in annot_db.columns:
                    del annot_db['duration']

        col_names += [c for c in annot_db.columns.values if c not in col_names]
        annot_db[['record_id','annot_id']] = annot_db[['record_id','annot_id']].astype(numpy.int64)
        
        if db=='record_db': 
            iter_rec = iter(self.subset(select,by='record_db').subset("record_id>"+str(last_rid),by='record_db'))
        else:
            if db not in self.annot:
                raise Error("db should be either 'record_db' or a member of self.annot")  
            iter_rec = self.iter_annot(db,select,_select="record_id>"+str(last_rid),flanking=flanking)
            
        return (iter_rec, annot_db, col_names)
        
        
    def detect(self,db='record_db',annotate=None,select=None,channel='s',method='env',env_method='default',
               threshold=(0.04,0.01), max_ROI=100, augment=False, flanking=0, **kwargs):
        """
        detects regions of interest in the records of the warble object

        db - str
            name of the DataFrame on which the matching will be done
            may be either 'record_db' or a member of self.annot
        annotate - str or None
            name of the annotation DataFrame to be created in self.annot
            if none, the function returns the annotation dataframe without registering it 
        select - str or pandas.Series of bools
            selection of db
        channel - string
            indicates the channel of the each record that will be used for the matching
        method - str or callable
            Indicates the method used to find the regions of interest
            'env' calculates a smooth envelope and uses a threshold
            'all' detects all as a single ROI
            if callable should accept a Rec and return an Env
        env_method - str or function
            method to calculate de envelope. If it is a function it should accept a Wav and return an Env.
            if 'default' is selected, a Jarne envelope with bin_size=4415, cut_freq=5, padtype=None is used
        threshold - float or 2-tuple of floats
            upper/lower threshold for the segmentation
        max_ROI - int
            maximun number of regions of interest in a record
        augment - bool
            indicated if match should augment previous runs or start from scratch
            if True, it will only analyze records or annotatios woth record_id larger than the 
            last record_id of the annotation DataFrame specified in 'annotate'
        flanking - float
            time in seconds by which to flank the annotation (used if db is in self.annot)
        kwargs - dict
            further arguments for envelope method. 
            
        If annotate is None, returns annotation DataFrame with columns 'record_id', 'annot_id', 'start', 'end', 'comment'. 
        If annotate is a str, returns a modified warble object with the registered annotations DataFrame 
        """
        
        (iter_rec, annot_db, col_names) = self._init_annot(db,annotate,select,augment,flanking)
        
        if len(iter_rec)==0:
            warnings.warn("No record to annotate")
            if annotate is None:
                return annot_db
            else:
                return self.register_annot(annotate,annot_db)
        
        #setting detection method
        if callable(method):
            detect_fun=lambda rec: Annotate.detect_env(method(rec),threshold=threshold,max_ROI=max_ROI)
        elif isinstance(method,str):
            if method == 'env':
                if env_method=='default':
                    if 'bin_size' not in kwargs: kwargs['bin_size']=4415
                    if 'cut_freq' not in kwargs: kwargs['cut_freq']=5
                    if 'padtype' not in kwargs: kwargs['padtype']=None
                    calc_env = lambda wav: wav.envelope(method='jarne',**kwargs)
                elif isinstance(env_method,str):
                    calc_env = lambda wav: wav.envelope(method=env_method,**kwargs)
                elif callable(env_method):
                    calc_env = env_method
                else:
                    raise Error("env_method shlould be a string or callable")
                detect_fun = lambda rec: Annotate.detect_env(calc_env(rec[channel]),threshold=threshold,max_ROI=max_ROI)
            elif method == 'all':
                detect_fun = lambda rec: Annotate.detect_all(rec[channel])
            else:
                raise Error("Unknown mehtod")
        else:
            raise Error("method should be callable or str")
                
        for rec in progressbar(iter_rec):
            #rec=w[11]
            try:
                annot_rec=detect_fun(rec)
                annot_rec.insert(0,'record_id',int(rec.rid))
                annot_db=annot_db.append(annot_rec, ignore_index=True, verify_integrity=True)
            except Exception as err:
                warnings.warn("Record "+str(rec.rid)+" failed"+"\n"+str(err))

        col_names += [c for c in annot_db.columns.values if c not in col_names]
        #annot_db=annot_db.reindex_axis(col_names,axis=1)
        annot_db=annot_db.reindex(col_names,axis=1)
        annot_db.sort_values(['record_id','start'],inplace=True) 
        annot_db['annot_id']=annot_db.groupby('record_id')['annot_id'].transform(lambda x:numpy.arange(len(x)))
        annot_db.set_index(['record_id','annot_id'],drop=False,inplace=True,verify_integrity=True)
        annot_db.index.rename(['record_idx','annot_idx'],inplace=True)
        
        if annotate is not None:
            return self.annot_register(annotate,annot_db).invisible()
        else:
            return annot_db
            
    def match(self,db='record_db',to=None,annotate=None,select=None,channel='s',augment=False,
              flanking=0, preprocess=None, **kwargs):
        """
        Match selected records or annotations to a template
        
        db - str
            name of the DataFrame on which the matching will be done
            may be either 'record_db' or a member of self.annot
        to - Wav, Env or Spe object, or callable 
            templated to be matched to
            if 'to' is callable, it should accept a record or annotation and return a suitable template (Wav, Env or Spe)
        annotate - str or None
            name of the annotation DataFrame to be created in self.annot
            if none, the function returns the annotation dataframe without registering it 
        channel - string
            indicates the channel of the each record that will be used for the matching
        select - str of pandas.Series of boolen 
            selection of db
        augment - bool
            indicated if match should augment previous runs or start from scratch
            if True, it will only analyze records or annotatios woth record_id larger than the 
            last record_id of the annotation DataFrame specified in 'annotate'
        flanking - float
            time in seconds by which to flank the annotation (used if db is in self.annot)
        preprocess - None or callable
            if callable, it should take a Wav and return a modified Wav. 
        kwargs - dictionary
            further arguments for Annotate.match_obj
            
        If annotate is None, returns annotation DataFrame with columns 'record_id', 'annot_id', 'start', 'end', 'comment'. 
        If annotate is a str, returns a modified warble object with the registered annotations DataFrame 
        """
        
        if type(to).__name__ not in ['Wav','Env','Spe','function']:
            raise Error("'to' should be a template of type Wav, Env, Spe or a function")
            
        #col_names=['record_id','annot_id','start','end','comment']
        #annot_db = pandas.DataFrame(columns=col_names) 
        #if db=='record_db': 
        #    iter_rec = iter(self.subset(select,by='record_db'))
        #else:
        #    if db not in self.annot:
        #        raise Error("db should be either 'record_db' or a member of self.annot")  
        #        iter_rec = self.iter_annot(db,select,flanking=flanking)
  
        (iter_rec, annot_db, col_names) = self._init_annot(db,annotate,select,augment,flanking)
            
        if len(iter_rec)==0:
            warnings.warn("No record to annotate")
            if annotate is None:
                return annot_db
            else:
                return self.register_annot(annotate,annot_db)
        
        if preprocess is None:
            preprocess = lambda x:x
    
        for rec in progressbar(iter_rec):
            try:
                wav=preprocess(rec[channel])
                if callable(to):
                    pattern = to(rec)
                else:
                    pattern = to
                cdb = Annotate.match_obj(wav,pattern,**kwargs)
                cdb.insert(0,'record_id',int(rec.rid))
                annot_db=annot_db.append(cdb, ignore_index=True, verify_integrity=False)
            except Exception as err:
                #pdb.set_trace()
                warnings.warn("Record "+str(rec.rid)+" failed with error type "+str(type(err))+"\n"+str(err))
            
        col_names += [c for c in annot_db.columns.values if c not in col_names]
        #annot_db=annot_db.reindex_axis(col_names,axis=1)
        annot_db=annot_db.reindex(col_names,axis=1)
        annot_db.sort_values(['record_id','start'],inplace=True) 
        annot_db['annot_id']=annot_db.groupby('record_id')['annot_id'].transform(lambda x:numpy.arange(len(x)))
        annot_db[['record_id','annot_id']] = annot_db[['record_id','annot_id']].astype(numpy.int64)
       
        if annotate is not None:
            return self.annot_register(annotate,annot_db).invisible()
        else:
            return annot_db
            
            
    def consolidate(self,db,when,additive='all',annotate=None,select=None,augment=False):
        """
        Consolidate an annotation data base. 
        
        db - str
            name of the DataFrame to consolidate. A member of self.annot
        when - str or callable
            criteria to define when to consolidate two or more annotations
            The criteria should evaluate to True or False on the candidate consolidated annotation row
            if callable, should accept a pandas.Series and return a boolean
            if str, should be a condition that evaluates to True or False when evaluated on a pandas.Series. 
        additive - list of str or 'all'
            indicates which variables should be treated as 'additive'.
            If 'all', all the variables not required will be treated as additive
        annotate - str or None
            name of the annotation DataFrame to be created in self.annot
            if none, the function returns the annotation dataframe without registering it 
        select - str or pandas.Series of bools
            selection of db
        augment - bool
            indicated if match should augment previous runs or start from scratch
            if True, it will only analyze records or annotatios woth record_id larger than the 
            last record_id of the annotation DataFrame specified in 'annotate'
        """
        consolidated_db=pandas.DataFrame() 
        last_rid=0        
        if annotate is not None:
            if annotate in self.annot and augment is True:
                consolidated_db = self.annot[annotate]
                last_rid=consolidated_db.iloc[-1].record_id

        if db not in self.annot:
            raise Error("db should be either 'record_db' or a member of self.annot")  
        annot_db=self.subset(select,by=db).subset("record_id>"+str(last_rid),by='record_db').annot[db]
        new_consolidated_db=Annotate.consolidate(annot_db,when,additive)
        consolidated_db=pandas.concat([consolidated_db,new_consolidated_db],ignore_index=True)
        consolidated_db.set_index(['record_id','annot_id'],drop=False,inplace=True,verify_integrity=True)
        consolidated_db.index.rename(['record_idx','annot_idx'],inplace=True)
        
        if annotate is not None:
            return self.annot_register(annotate,consolidated_db).invisible()
        else:
            return consolidated_db     

    def annot_difference(self,x_name,y_name,annotate=None,x_cols='all'):
        """
        Calculates the difference between two annotation dbs
        
        x_name - str
            name of the first annotation db used in the difference. Should be a member of self.annot
            (annotations of y_name will be substracted to x_name's annotations)
        y_name - str
            name of the second annotation db used in the difference. Should be a member of self.annot
            (annotations of y_name will be substracted to x_name's annotations)
        annotate - str or None
            name of the annotation DataFrame to be created in self.annot
            if none, the function returns the annotation dataframe without registering it 
        x_cols - str, list of str or None
            columns of x_name to be included in the final DataFrame, with x_name_ as prefix
            Special values None and 'all' do exactly that.
            
        return the modified Warble object if annotate is not None, and a DataFrame if annotate is None
        """
        if x_name not in self.annot:
            raise Error("x_name should be a member of self.annot")
        if y_name not in self.annot:
            raise Error("y_name should be a member of self.annot")

        x_db=self.annot[x_name]
        x_db.name=x_name
        y_db=self.annot[y_name]            
        y_db.name=y_name            

        if x_cols=='all':
            x_cols=list(set(x_db.columns.values)-set(['record_id','annot_id']))
        elif x_cols is None:
            x_cols=[]
        elif not isinstance(x_cols,list):
            x_cols=[x_cols]
            
        diff_db=Annotate.difference(x_db,y_db)

        if len(x_cols)>0:
            x_db=x_db[x_cols].rename(columns={col:x_name+"_"+col for col in x_cols})
            diff_db=diff_db.join(x_db,on=['record_id',x_name+'_annot_id'])
            
        if annotate is not None:
            return self.annot_register(annotate,diff_db).invisible()
        else:
            return diff_db                 
            
    def annot_intersect(self,x_name,y_name,annotate=None,x_cols='all',y_cols='all'):
        """
        Calculates the intersectin between two annotation dbs
        
        x_name - str
            name of the first annotation db used in the intersection. Should be a member of self.annot
        y_name - str
            name of the second annotation db used in the intersection. Should be a member of self.annot
        annotate - str or None
            name of the annotation DataFrame to be created in self.annot
            if none, the function returns the annotation dataframe without registering it 
        x_cols - str, list of str or None
            columns of x_name to be included in the final DataFrame, with x_name_ as prefix
            Special values None and 'all' do exactly that.
        y_cols - str, list of str or None
            columns of y_name to be included in the final DataFrame, with y_name_ as prefix
            Special values None and 'all' do exactly that.
            
        return the modified Warble object if annotate is not None, and a DataFrame if annotate is None
        """
        
        if x_name not in self.annot:
            raise Error("x_name should be a member of self.annot")
        if y_name not in self.annot:
            raise Error("y_name should be a member of self.annot")

        x_db=self.annot[x_name]
        x_db.name=x_name
        y_db=self.annot[y_name]            
        y_db.name=y_name            

        if x_cols=='all':
            x_cols=list(set(x_db.columns.values)-set(['record_id','annot_id']))
        elif x_cols is None:
            x_cols=[]
        elif not isinstance(x_cols,list):
            x_cols=[x_cols]
            
        if y_cols=='all':
            y_cols=list(set(y_db.columns.values)-set(['record_id','annot_id']))
        elif y_cols is None:
            y_cols=[]
        elif not isinstance(y_cols,list):
            y_cols=[y_cols]
            
        intersect_db=Annotate.intersect(x_db,y_db)
        if len(x_cols)>0:
            x_db=x_db[x_cols].rename(columns={col:x_name+"_"+col for col in x_cols})
            intersect_db=intersect_db.join(x_db,on=['record_id',x_name+'_annot_id'])
        if len(y_cols)>0:
            y_db=y_db[y_cols].rename(columns={col:y_name+"_"+col for col in y_cols})
            intersect_db=intersect_db.join(y_db,on=['record_id',y_name+'_annot_id'])

        if annotate is not None:
            return self.annot_register(annotate,intersect_db).invisible()
        else:
            return intersect_db     
        
            
    def get_ensemble(self, annot='record', get=lambda rec: rec.s, select=None, set_delay=0, flanking=0, warped_delays=False):
        """
        Gets an ensemble from a warble object
        
        An ensemble is a matrix where each row is an annotation and 
        each column is a time. 
        
        annot - str
            name of the annotation dataframe from which to take the annotations
        get - extraction function
            should accept a record and return a Wav or Env object
        select - str or pandas.Series
            selection criteria for the annotations
        set_delay - float or None
            delay to be set for each Wav resulting from get function. 
            If None the delay (or start) of the annotations are not changed.
        flanking - float
            time in seconds by which to flank the annotation
        warped_delays - Bool
            if True and elements inherit from class Warped a warped_delays attributed is
            added to the output Ens object. This attribute contains an Ens object of the
            same dimensions as the original containing the warped delays as returned by 
            Warped.get_delays_env().
            
        returns an object of type Ens
        """
        if not annot in self.annot:
            raise Error(str(annot)+" not in warble.annot")
        db=self.annot[annot].copy()    
        #filtering
        if select is not None and len(db)>0:
            if type(select)==str: 
                db.query(select,inplace=True)
            else:
                db=db.loc[select,:]       
 
        ens_list=[get(ann).set_delay(set_delay) for \
                      ann in progressbar(self.iter_annot(annot,select=select,flanking=flanking))]
 
        ens_duration=[ens_v.duration for ens_v in ens_list]
        max_dur_wav=ens_list[ens_duration.index(max(ens_duration))]
        times=max_dur_wav.get_times() 
        
        ens_matrix=numpy.matrix([ens_v.interpolate(times) for ens_v in ens_list])
        ensemble=Ens(ens_matrix, db.index, times=times, tjust=max_dur_wav.tjust)
        ensemble.db=db
        ensemble.annot_type=annot

        if warped_delays and isinstance(ens_list[0], Warped):
            if set_delay is not None:
                warnings.warn('set_delay should be None when using warped_delays')
            ens_warped_delays = numpy.matrix([ens.get_delays_env().interpolate(times) for ens in ens_list])
            ensemble.warped_delays = Ens(ens_warped_delays, db.index, times=times, tjust=max_dur_wav.tjust)       

        return ensemble
            
    def invisible(self):
        self.is_invisible=True
        return self
    
    def __getitem__(self, key):
        """
        Get record by positinal index in record_db
        """
        return self.get_record(self.record_db.record_id.iloc[key])
        
    def __len__(self):
        return self.record_db.shape[0]
        
    def __contains__(self, record):
        return record.record['record_id'] in self.record_db['record_id']
    
    def __getattr__(self, attr):
        if attr in self.record_db.columns:
            return self.record_db[attr] 

    def __deepcopy__(self, memo):
        return Warble(**copy.deepcopy(self.__dict__))

    def __repr__(self):
        if not self.is_invisible:
            s="%s(%i record(s), channel(s): "%(self.__class__.__name__, self.record_db.shape[0])
            for ch in self.channels:
                if pandas.notnull(ch): s+=str(ch)+", "
            if self.lazy_load: s=s+"lazy_load=True, "
            s=s[0:-2]+")\n"
            if len(self.preprocess) > 0:
                s+="preprocess channels "            
                for ch in self.preprocess:
                    s+=str(ch)+", "
                s=s[0:-2]+"\n"
            s+="Member DataFrame(s):\n"
            for db in self.__dict__:
                if isinstance(getattr(self,db),pandas.DataFrame):
                    s+="\t"+db+": "    
                    #for var in getattr(self,db).columns: s+=str(var)+", "
                    s+=numpy.array2string(getattr(self,db).columns,max_line_width=100,separator=",",\
                                          formatter={'str_kind':lambda x: " %s" % x})[1:-1].\
                                          replace("\n","\n\t\t")
                    s+="\n"
            if len(self.annot) > 0:
                s+="Annotation DataFrame(s):\n"
                for annot_type in self.annot:
                    s+="\t"+annot_type+"[N="+str(len(self.annot[annot_type]))+"]: "    
                    #for var in self.annot[annot_type].columns: 
                    #    s+=str(var)+", "
                    s+=numpy.array2string(self.annot[annot_type].columns,max_line_width=100,separator=",",\
                                          formatter={'str_kind':lambda x: " %s" % x})[1:-1].\
                                          replace("\n","\n\t\t")
                    s+="\n"
            s+="Files in path(s): \n"
            for path in self.file_db.path.unique(): s+="\t"+path+"\n"
            return s
        elif self.is_invisible:
            self.is_invisible=False
            return ""
        
    def __iter__(self):
        return rec_iter(self,list(self.record_db.record_id))

        
        
    @property
    def db(self):
        if (self.record_db is None) or (not 'record_id' in self.record_db):
            return None
        else:
            return self.record_db

    @property
    def channels(self):
        return self.file_db.channel.unique().tolist()
            
class rec_iter(collections.abc.Iterator):
    def __init__(self,warble,rec_seq):
        self.warble=warble
        self.rec_seq=list(rec_seq)
    def __iter__(self):
        return(self)
    def __next__(self):
        if len(self.rec_seq)>0:
            return self.warble.get_record(self.rec_seq.pop(0))
        else:
            raise StopIteration
    def __len__(self):
        return len(self.rec_seq)
      
class ann_iter_warble(collections.abc.Iterator):
    def __init__(self,warble,annot_db,annot_type,flanking=0):
        self.warble=warble
        self.annot_db=annot_db
        self.annot_type=annot_type
        self.flanking=flanking
        self.idx=0
        if len(self.annot_db)>0:
            self.last_rec=warble.get_record(annot_db.iloc[0].record_id)
    def __iter__(self):
        return(self)
    def __next__(self):
        if self.idx<len(self.annot_db):
            rid=self.annot_db.iloc[self.idx].record_id
            if not rid==self.last_rec.rid:
                self.last_rec=self.warble.get_record(rid)
            ann=self.last_rec.get_annot(self.annot_db.iloc[self.idx],annot_type=self.annot_type,flanking=self.flanking)
            self.idx+=1
            return ann
        else:
            raise StopIteration
    def __len__(self):
        return len(self.annot_db)
        
        
def progressbar(iter_):
    """
    Simple progress bar utility
    """
    i=1; N_pts=0
    N=len(iter_)
    if N==0:
        return
    N_pb=int(math.ceil(N/100))
    msg="\n| 0% - Calculating "+str(N)+" values"
    print(msg+" "*(94-len(msg))+"100% |",end="\n",flush=False)
    yield next(iter_)
    while True:
        try:
            i+=1
            if i%N_pb==0:
                N_new=int(math.floor(100*i/N)-N_pts)
                print('\b',"."*N_new,sep="",end=" ",flush=N_pts>5)
                N_pts+=N_new
            if i==N-1:
                print(end="\n",flush=True)
            yield next(iter_)
        except StopIteration:
            return

    