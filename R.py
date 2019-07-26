#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R.py module of WarblePy. Small module to facilitate working with rpy2.

@author: Alan Bush
"""

import rpy2.robjects as ro; R = ro.r
import rpy2.robjects.lib.ggplot2 as gg
import rpy2.robjects.lib.grid as grid
from rpy2.robjects import pandas2ri; pandas2ri.activate()
from rpy2.robjects.packages import importr
import numpy as np

#utils = importr("utils")
base = importr('base')
grDevices = importr("grDevices")

NULL=R('NULL')
c=R.c
aes=gg.aes_string; 

class Error(Exception): pass

def df2R(df,as_factor=None,int64_to_str=True):
    """
    Transforms a pandas dataframe into an R dataframe. 
    """
    if int64_to_str is True:
        df=df.copy()
        for col in df.columns:
            if df[col].dtype==np.int64:
                df[col]=df[col].astype(str)
    
    dfR=pandas2ri.py2ri(df)
    if as_factor is not None:
        if type(as_factor) is str:
            as_factor={as_factor:np.sort(df[as_factor].unique())}
        elif type(as_factor) is list:
            as_factor={f:np.sort(df[f].unique()) for f in as_factor}
        elif type(as_factor) is dict:
            as_factor = {**as_factor, **{f:np.sort(df[f].unique()) for f in as_factor if as_factor[f] is None}}
        else:
            raise Error('as_factor should be str, list or dict')
        for f in as_factor:
            idx=[i for i,s in enumerate(R.names(dfR)) if s==f][0]
            dfR[idx]=ro.vectors.FactorVector(dfR[idx],levels=ro.vectors.StrVector(as_factor[f]),ordered=True)
            
    for i in range(df.shape[1]):
        if df.dtypes[i].name == 'category':
            if df.iloc[:,i].cat.ordered:
                dfR[i]=ro.vectors.FactorVector(dfR[i],levels=ro.vectors.StrVector(df.iloc[:,i].cat.categories),ordered=True)
            
    return dfR
            
