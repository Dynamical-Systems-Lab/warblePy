#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility fucntions for WarblePy.

@author: Alan Bush
"""
import os
import copy
import warnings
import Warble
import Wav
import Rec
import tempfile

def praat(*args, max_files=30):
    """
    Open Wav, Rec or Warble objects in praat 
    
    *args - objects to be loaded in praat
    
    max_files - integer
        max number of files to open
    """
    files = list()
    def wav2path(wav,prefix='tmp'):
        assert isinstance(wav,Wav.Wav)
        if wav.file_path=='' or wav.file_name=='':
            f=tempfile.NamedTemporaryFile(mode='w', suffix='.wav', prefix=prefix+"__", delete=False)
            wav.write(f.name,log=None)
            wav.file_path=os.path.dirname(f.name)
            wav.file_name=os.path.basename(f.name)
        return os.path.join(wav.file_path,wav.file_name)
        
    for arg in args:
        if isinstance(arg,Warble.Warble):
            if len(arg) > max_files: warnings.warn("Warble object contains more records than max_files")
            for i in range(0,min(max_files,len(arg))):
                rec=arg[i].zero_pad()
                for ch in rec: files.append(wav2path(rec[ch],ch))
        elif isinstance(arg,Rec.Rec):
            rec=arg.zero_pad()
            for ch in rec: files.append(wav2path(rec[ch],ch))
        elif isinstance(arg,Wav.Wav):
            files.append(wav2path(arg))
        elif isinstance(arg,str):
            files.append(arg)
        else:
            warn="Unknown type: " + str(type(arg))
            warnings.warn(warn)
    if len(files) > max_files: 
        warnings.warn("More files than max_files. Opening first "+max_files+" files.")
    cmd="praat --open "
    for file in files[0:min(len(files),max_files)]: cmd+=file+" "
    cmd+="&"
    os.system(cmd)
        
def extract_pb_from_rec(rec): 
    """
    Extracts playback from record. Modifies input. 
    """
    assert "pb_s" in rec
    assert "pb_vS" in rec
    rec["s"]=rec["pb_s"]
    rec["vS"]=rec["pb_vS"]
    rec.vS.delay=0
    rec.s.delay=0
    rec.record=None
    del rec["pb_s"]
    del rec["pb_vS"]
    return rec
    
    
    
    