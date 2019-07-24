#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NML.py module of WarblePy. 

Provides functions to perform Non-Negative Matrix Factorization

Author: Alan Bush 
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
from Spe import Ens

class Error(Exception): pass


def NMF_ISRA_lag(Y, A=None, X=None, J=None, S=0, maxIter=1000, alpha_A=0, alpha_X=0)
    """
    ISRA algorithm for Nonnegative Matrix Factorization (NMF)
            
    INPUT
    Y - Matrix of I x T representing data 
    A - Matrix of I x J representing weights of each gesture in each instance
    X - Matrix of J x T representing each gesture at each time
    
    J - Rank of the decomposition (number of gestures to use)
        Defaults to 10 if X not given
    S - number of Lag samples
    maxIter - max number of iterations.
    alpha_A - sparsness parameter for A.     
    alpha_X - sparsness parameter for X.     
       
    OUTPUT
    A, X and L such that || Y − sum(((L==s) * A) @ X^[s->],{s,-S,S}) || is minimized
    
    This function was inspired by code by Anh Huy Phan anh Andrzej Cichocki 2008
    """

    eps = np.finfo(float).eps

    # Forcing Y to positive matrix
    Y = np.nan_to_num(Y)
    Y[Y <= 0] = eps
    (I,T) = np.shape(Y)

    #checkign or initializing inputs
    if X is None:
        if J is not None:
            X = np.random.rand(J,T)
            X[X<=0] = eps
        else:
             raise Error("J not defined")
    else:
        X = np.nan_to_num(X)
        X[X<=0] = eps
        (J1,T1) = np.shape(X)
        if J is not None and J != J1:
            raise Error("J inconsistent with X")
        if T != T1:
            raise Error("dimension of X inconsistent with that of Y")
        J=J1
 
    if A is None:
        A = np.random.rand(I,J)
        A[A<=0] = eps
    else:
        A = np.nan_to_num(A)
        A[A<0] = eps
        (I2,J2) = np.shape(A)
        if J != J2:
            raise Error("A inconsistent with J or X")
        if I != T2:
            raise Error("dimension of A inconsistent with that of Y")
    
    L = np.zeros((I,J))
        
    for k in range(1,maxIter):
        
        Yhat = np.zeros((I,T))
        for s in range(-S,S+1):
            Yhat = Yhat + ((L==s)*A) @ np.roll(X,s,axis=1)
        
        U = np.array([ (Y @ np.transpose(np.roll(X,s))) / ((Yhat @ np.transpose(np.roll(X,s))) + alpha_A)
                for s in range(-S,S+1)])
        
        ssUs = np.zeros((I,J))
        for s in range(-S,S+1):
            ssUs = ssUs + (L==s) * U[s,:,:]
        A = A * ssUs

        L = L + np.sign(np.argmax(U,axis=0) - S - L)
        
        #X = np.clip(X * (np.transpose(A) @ Y) / ((np.transpose(A) @ Yhat) + alpha_X + eps),eps,16)        
        
        #Spe.Ens(X,y=None,times=ens.times).plot()

i=1
#i=np.random.randint(0,I)
plt.plot(Yhat[i,:])
plt.plot(Y[i,:])


alpha_A=300
alpha_X=0

A[i,:]
L[i,:]

X = G.array
G.plot()

plt.hist(A.flatten(),100)










































def NMF_ISRA(Y, A=None, X=None, J=None, maxIter=1000, alpha_A=0, alpha_X=0)
    """
    ISRA algorithm for Nonnegative Matrix Factorization (NMF)
            
    INPUT
    Y - Matrix of I x T representing data 
    A - Matrix of I x J representing weights of each gesture in each instance
    X - Matrix of J x T representing each gesture at each time
    
    J - Rank of the decomposition (number of gestures to use)
        Defaults to 10 if X not given
    maxIter - max number of iterations.
    alpha_A - sparsness parameter for A.     
    alpha_X - sparsness parameter for X.     
       
    OUTPUT
    A and X such that ||Y − A ∗ X|| is minimized
    
    This function was inspired by code by Anh Huy Phan anh Andrzej Cichocki 2008
    """

    eps = np.finfo(float).eps

    # Forcing Y to positive matrix
    Y = np.nan_to_num(Y)
    Y[Y <= 0] = eps
    (I,T) = np.shape(Y)

    #checkign or initializing inputs
    if X is None:
        if J is not None:
            X = np.random.rand(J,T)
            X[X<=0] = eps
        else:
             raise Error("J not defined")
    else:
        X = np.nan_to_num(X)
        X[X<=0] = eps
        (J1,T1) = np.shape(X)
        if J is not None and J != J1:
            raise Error("J inconsistent with X")
        if T != T1:
            raise Error("dimension of X inconsistent with that of Y")
        J=J1
 
    if A is None:
        A = np.random.rand(I,J)
        A[A<=0] = eps
    else:
        A = np.nan_to_num(A)
        A[A<0] = eps
        (I2,J2) = np.shape(A)
        if J != J2:
            raise Error("A inconsistent with J or X")
        if I != T2:
            raise Error("dimension of A inconsistent with that of Y")
       
    for k in range(1,maxIter):
        Yhat = A @ X
        
        A = np.clip(A * (Y @ np.transpose(X)) / ((Yhat @ np.transpose(X)) + alpha_A + eps),eps,2)
        
        #X = np.clip(X * (np.transpose(A) @ Y) / ((np.transpose(A) @ Yhat) + alpha_X + eps),eps,16)        
        
        #Spe.Ens(X,y=None,times=ens.times).plot()

