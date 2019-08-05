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

import itertools
import numpy as np
import matplotlib.pyplot as plt

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
    
    
def scatterplot_matrix(data, names, lim=[0,2], **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    
        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[x,y].plot(data[y], data[x], **kwargs)


    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig

    
    