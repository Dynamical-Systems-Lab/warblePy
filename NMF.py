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

class Error(Exception): pass

def NMF_ISRA_lag(Y, A=None, X=None, L=None, S=0, J=None, \
                 fit_A=None, fit_X=None, fit_L=None, \
                 A_max=None, A_min=None, A_update_factor_lim=[0.7071,1.4142], 
                 X_max=None, X_min=None, X_update_factor_lim=[0.7071,1.4142],
                 alpha_A=0, alpha_X=0,  \
                 maxIter=1000, convergence_threshold=1e-3):
    """
    Nonnegative Matrix Factorization (NMF) with lags.
    Implemented using Lee-Seung type Image Space Reconstruction Algorithm (ISRA) 
            
    INPUT
    Y - Matrix of I x T representing data 
    A - Matrix of I x J representing weights of each gesture in each instance
    X - Matrix of J x T representing each gesture at each time
    L - Integer matrix of I x J repesenting 'lags' of each component in samples
    
    S - number of Lag samples. If 2-tuple interpreted as (min, max)
    J - rank of the decomposition. Required if X is not given

    fit_A - bool indicating if A should be fitted. Defaults to False if A is given.
    fit_X - bool indicating if X should be fitted. Defaults to False if X is given.
    fit_L - bool indicating if L should be fitted. Defaults to False if L is given.

    A_max - max value for elements of A
    A_min - minimal value to be considered different from zero after optimization
    X_max - max value for elements of X
    X_min - minimal value to be considered different from zero after optimization
    A_update_factor_lim - list of length 2 given lower and upper bound for update factors
    X_update_factor_lim - list of length 2 given lower and upper bound for update factors
    
    alpha_A - sparsness parameter for A.  
    alpha_X - sparsness parameter for A.  
    
    maxIter - max number of iterations.
    convergence_threshold - fractional change of convergence between successive iterations
       
    OUTPUT
    A, X and L such that 
    
        D = 0.5*|| Y − Yhat||_2^2 + alpha_A*||A||_1 + alpha_X*||X||_1 
    
    where
    
        Yhat = sum(((L==s) * A) @ np.roll(X,s) for s in range(-S,S+1))
    
    is minimized
    
    RETURNS
    (A,X,L,Yhat,D)
    
    This function was inspired by code by Anh Huy Phan anh Andrzej Cichocki 2008
    """

    #checkign or initializing inputs
    (I,T) = np.shape(Y)

    if X is None:
        if J is None:
            raise Error('X or J must be specified')
        X = np.random.rand(J,T)
        if fit_X is None:
            fit_X=True
    else:
        X = np.nan_to_num(X)
        (J1,T1) = np.shape(X)
        if T1 != T:
            raise Error("dimension of X inconsistent with that of Y")
        if J is None:
            J = J1
        elif J!=J1:
            raise Error('J is inconsistent with shape of X')    
        if fit_X is None:
            fit_X=False
         
    if A is None:
        A = np.random.rand(I,J)
        if fit_A is None:
            fit_A = True
    else:
        A = np.nan_to_num(A)
        (I2,J2) = np.shape(A)
        if J2 != J:
            raise Error("dimension of A inconsistent with J or X")
        if I2 != I:
            raise Error("dimension of A inconsistent with that of Y")
        if fit_A is None:
            fit_A = False

    if L is None:
        L = np.zeros((I,J),dtype=int)   
        if fit_L is None:
            fit_L = True
    else:
        L = np.nan_to_num(L)
        (I3,J3) = np.shape(L)    
        if J3 != J:
            raise Error("dimension of L inconsistent with J or X")
        if I3 != I:
            raise Error("dimension of L inconsistent with that of Y")
        if fit_L is None:
            fit_L = False
    
    if len(S) == 1:
        S_min = int(-abs(S))
        S_max = int(abs(S))
    elif len(S) == 2:
        S_min = int(S[0])
        S_max = int(S[1])
    else:
        raise Error('S should have 1 or 2 elements')    

    # Forcing Y to positive matrix
    eps = np.finfo(float).eps
    Y = np.nan_to_num(Y)
    Y[Y <= 0] = eps
    X[X<=0] = eps 
    A[A<=0] = eps

    dp1X = np.roll(X,+1)-X
    dm1X = np.roll(X,-1)-X
    IJx1 = np.ones((I,J))
    M = IJx1 @ ((dp1X @ dp1X.T) * np.identity(J))

    count = 0
    converged = False 
    Dlast = np.inf
    
    while count < maxIter and not converged:
        
        #updating Yhat
        Yhat = np.zeros((I,T))
        for s in range(S_min,S_max+1):
            Yhat = Yhat + ((L==s)*A) @ np.roll(X,s,axis=1)
        D = 0.5*np.linalg.norm(Y-Yhat)**2
        
        #updating A
        if fit_A:
            uA = np.zeros((I,J))
            for s in range(S_min,S_max+1):
                uA = uA + (L==s) * ((Y @ np.roll(X,s).T) / ((Yhat @ np.roll(X,s).T) + alpha_A + eps))
            uA = np.clip(uA, A_update_factor_lim[0], A_update_factor_lim[1])    
            A = np.clip(A * uA,0,A_max)

        #updating X
        if fit_X:
            uXnum = np.zeros((J,T))
            uXdenom = np.zeros((J,T)) 
            for s in range(S_min,S_max+1):
                uXnum = uXnum + ((L==s).T * A.T) @ np.roll(Y,-s)
                uXdenom = uXdenom + ((L==s).T * A.T) @ np.roll(Yhat,-s)
            uX = uXnum / (uXdenom + alpha_X + eps)
            uX = np.clip(uX, X_update_factor_lim[0], X_update_factor_lim[1]) 
            X = np.clip(X * uX,0,X_max) 

        # updating L
        if fit_L:
            delta_Lp1_D_overA = np.zeros((I,J))
            delta_Lm1_D_overA = np.zeros((I,J))
            for s in range(S_min,S_max+1):
                delta_Lp1_D_overA = delta_Lp1_D_overA + (L==s) * ((Y-Yhat) @ np.roll(dp1X,s).T)
                delta_Lm1_D_overA = delta_Lm1_D_overA + (L==s) * ((Y-Yhat) @ np.roll(dm1X,s).T)
            delta_Lp1_D_overA = -delta_Lp1_D_overA + 0.5*A*M     
            delta_Lm1_D_overA = -delta_Lm1_D_overA + 0.5*A*M     
            uL =  np.sign(delta_Lm1_D_overA*(delta_Lm1_D_overA<0) - delta_Lp1_D_overA*(delta_Lp1_D_overA<0))
            L = np.clip(L + uL,S_min,S_max)

            
        if (D < convergence_threshold or abs(D/Dlast-1) < convergence_threshold):
            converged = True
            print('Converged at %i iterations' % count)
       
        Dlast=D
        count = count+1
        
    if count == maxIter and not converged:
        print('Reached %i iterations without convergence' % maxIter)
    
    if A_min is not None and fit_A:
        A[A<A_min] = 0
    if X_min is not None and fit_X:
        X[X<X_min] = 0
        
    return (A,X,L,Yhat,D)     

















#
#
#
# 20190730
#
#def NMF_ISRA_lag_fix_X(Y, A=None, X=None, W=None, S=0, A_max=2, A_min=1e-2, maxIter=1000, alpha_A=0,  convergence_threshold=1e-3):
#    """
#    Nonnegative Matrix Factorization (NMF) Image Space Reconstruction Algorithm (ISRA) 
#    with lagged fixed X (lfX)
#            
#    INPUT
#    Y - Matrix of I x T representing data 
#    A - Matrix of I x J representing weights of each gesture in each instance
#    X - Matrix of J x T representing each gesture at each time [fixed]
#    W - Matrix of J x J represeting the wright of each gesture
#    
#    S - number of Lag samples. If 2-tuple interpreted as (min, max)
#    A_max - max value for elements of A
#    A_min - minimal value to be considered different from zero after optimization
#    maxIter - max number of iterations.
#    alpha_A - sparsness parameter for A.  
#    convergence_threshold - fractional change of convergence between successive iterations
#       
#    OUTPUT
#    A and L such that || Y − sum(((L==s) * A) @ X^[s->], {s,-S,S}) || is minimized
#    
#    RETURNS
#    (A,L,W,Yhat)
#    
#    This function was inspired by code by Anh Huy Phan anh Andrzej Cichocki 2008
#    """
#
#    eps = np.finfo(float).eps
#
#    # Forcing Y to positive matrix
#    Y = np.nan_to_num(Y)
#    Y[Y <= 0] = eps
#    (I,T) = np.shape(Y)
#
#    #checkign or initializing inputs
#    if X is None:
#        raise Error('X required')
#    else:
#        X = np.nan_to_num(X)
#        X[X<=0] = eps
#        (J,T1) = np.shape(X)
#        if T1 != T:
#            raise Error("dimension of X inconsistent with that of Y")
# 
#    if W is None:
#        W = np.identity(J)
#    else:
#        W = np.nan_to_num(W)
#        (J1,J2) = W.shape
#        if J1 != J or J2 != J:
#            raise Error("incorrect dimensions for W")
#        W = W * np.identity(J)    
#    
#    if A is None:
#        A = np.random.rand(I,J)
#        A[A<=0] = eps
#    else:
#        A = np.nan_to_num(A)
#        A[A<0] = eps
#        (I2,J2) = np.shape(A)
#        if J2 != J:
#            raise Error("A inconsistent with J or X")
#        if I2 != I:
#            raise Error("dimension of A inconsistent with that of Y")
#    
#    if len(S) == 1:
#        S_min = int(-abs(S))
#        S_max = int(abs(S))
#    elif len(S) == 2:
#        S_min = int(S[0])
#        S_max = int(S[1])
#    else:
#        raise Error('S should have 1 or 2 elements')    
#
#
#    L = np.zeros((I,J),dtype=int)
#    #dX = (np.roll(X,1)-np.roll(X,-1))/2
#    dp1X = np.roll(X,+1)-X
#    dm1X = np.roll(X,-1)-X
#    IJx1 = np.ones((I,J))
#
#    count = 0
#    converged = False 
#    prev_norm_uA = 1
#    prev_norm_uL = 1
#    prev_norm_uW = 1
#    stage = 0
#    uL = np.zeros((I,J))
#    uW = np.identity(J)
#    while count<maxIter and not converged:
#        
#        #updating Yhat
#        Yhat = np.zeros((I,T))
#        for s in range(S_min,S_max+1):
#            Yhat = Yhat + ((L==s)*A) @ W @ np.roll(X,s,axis=1)
#       
#        #updating A
#        if stage >= 0:
#            uA = np.zeros((I,J))
#            gradADA3 = alpha_A*np.clip(((3*A)-IJx1)*(A-IJx1),0,1)
#            for s in range(S_min,S_max+1):
#                uA = uA + (L==s) * ((Y @ np.roll(X,s).T @ W) / ((Yhat @ np.roll(X,s).T @ W) + gradADA3 + eps))
#            uA = np.clip(uA,0.7071,1.4142)    
#            A = np.clip(A * uA,0,A_max)
#
#        # updating L
#        if stage >= 1:
#            M = IJx1 @ ((W @ dp1X @ dp1X.T @ W) * np.identity(J))
#            delta_Lp1_D_overA = np.zeros((I,J))
#            delta_Lm1_D_overA = np.zeros((I,J))
#            for s in range(S_min,S_max+1):
#                delta_Lp1_D_overA = delta_Lp1_D_overA + (L==s) * ((Y-Yhat) @ np.roll(dp1X,s).T @ W)
#                delta_Lm1_D_overA = delta_Lm1_D_overA + (L==s) * ((Y-Yhat) @ np.roll(dm1X,s).T @ W)
#            delta_Lp1_D_overA = -delta_Lp1_D_overA + 0.5*A*M     
#            delta_Lm1_D_overA = -delta_Lm1_D_overA + 0.5*A*M     
#            uL =  np.sign(delta_Lm1_D_overA*(delta_Lm1_D_overA<0) - delta_Lp1_D_overA*(delta_Lp1_D_overA<0))
#            L = np.clip(L + uL,S_min,S_max)
#
#        #updating W
#        if stage >= 2:
#            uWnum = np.zeros((I,J))
#            uWdenom = np.zeros((I,J)) 
#            for s in range(S_min,S_max+1):
#                uWnum = uWnum + (L==s) * (Y @ np.roll(X,s).T)
#                uWdenom = uWdenom + (L==s) * (Yhat @ np.roll(X,s).T)
#            gradADA1 = 0.1*alpha_A*(IJx1 - 2*(A>1))
#            uW = np.identity(J)*np.clip((A.T @ uWnum) / (A.T @ (uWdenom + gradADA1)),0.9,1.1) 
#            A = np.clip(A @ np.linalg.inv(uW),0,A_max)
#            W = np.clip(W * uW,0,2) 
#            
#        if count%10 == 0: 
#            norm_uA = np.linalg.norm((uA - 1)*A)
#            norm_uL = np.linalg.norm((uL)*(L>S_min)*(L<S_max))
#            norm_uW = np.linalg.norm((uW-np.identity(J)))
#            #print('norm_uA=%f  norm_uL=%f   at %i iteration' % (norm_uA,norm_uL,count) )
#            if (norm_uA < convergence_threshold or abs(norm_uA/prev_norm_uA-1) < convergence_threshold) \
#                and (norm_uL < convergence_threshold or abs(norm_uL/prev_norm_uL-1) < convergence_threshold) \
#                and (norm_uW < convergence_threshold or abs(norm_uW/prev_norm_uW-1) < convergence_threshold):
#                if stage > 2:
#                    converged = True
#                    print('Converged at %i iterations' % count)
#                else:
#                    stage = stage + 1    
#                    print('stage %i at iteration %i' % (stage, count))
#            prev_norm_uA=norm_uA
#            prev_norm_uL=norm_uL
#            prev_norm_uW=norm_uW
#            
#        count = count+1
#        
#    A[A<A_min] = 0
#    return (A,L,W,Yhat)     
#

















#20190726
#
#def NMF_ISRA_lag_fix_X(Y, A=None, X=None, S=0, A_max=1.3, A_min=1e-2, maxIter=1000, alpha_A=0,  convergence_threshold=1e-3):
#    """
#    onnegative Matrix Factorization (NMF) Image Space Reconstruction Algorithm (ISRA) 
#    with lagged fixed X (lfX)
#            
#    INPUT
#    Y - Matrix of I x T representing data 
#    A - Matrix of I x J representing weights of each gesture in each instance
#    X - Matrix of J x T representing each gesture at each time [fixed]
#    
#    S - number of Lag samples. If 2-tuple interpreted as (min, max)
#    A_max - max value for elements of A
#    A_min - minimal value to be considered different from zero after optimization
#    maxIter - max number of iterations.
#    alpha_A - sparsness parameter for A.  
#    convergence_threshold - fractional change of convergence between successive iterations
#       
#    OUTPUT
#    A and L such that || Y − sum(((L==s) * A) @ X^[s->], {s,-S,S}) || is minimized
#    
#    RETURNS
#    (A,L,Yhat)
#    
#    This function was inspired by code by Anh Huy Phan anh Andrzej Cichocki 2008
#    """
#
#    eps = np.finfo(float).eps
#
#    # Forcing Y to positive matrix
#    Y = np.nan_to_num(Y)
#    Y[Y <= 0] = eps
#    (I,T) = np.shape(Y)
#
#    #checkign or initializing inputs
#    if X is None:
#        raise Error('X required')
#    else:
#        X = np.nan_to_num(X)
#        X[X<=0] = eps
#        (J,T1) = np.shape(X)
#        if T1 != T:
#            raise Error("dimension of X inconsistent with that of Y")
# 
#    if A is None:
#        A = np.random.rand(I,J)
#        A[A<=0] = eps
#    else:
#        A = np.nan_to_num(A)
#        A[A<0] = eps
#        (I2,J2) = np.shape(A)
#        if J2 != J:
#            raise Error("A inconsistent with J or X")
#        if I2 != I:
#            raise Error("dimension of A inconsistent with that of Y")
#    
#    if len(S) == 1:
#        S_min = int(-abs(S))
#        S_max = int(abs(S))
#    elif len(S) == 2:
#        S_min = int(S[0])
#        S_max = int(S[1])
#    else:
#        raise Error('S should have 1 or 2 elements')    
#
#
#    L = np.zeros((I,J),dtype=int)
#    #dX = (np.roll(X,1)-np.roll(X,-1))/2
#    dp1X = np.roll(X,+1)-X
#    dm1X = np.roll(X,-1)-X
#    M = np.ones((I,J)) @ ((dp1X @ dp1X.T) * np.identity(J))
#
#    count = 0
#    converged = False 
#    prev_norm_uA = 1
#    prev_norm_uL = 1
#    while count<maxIter and not converged:
#        
#        Yhat = np.zeros((I,T))
#        for s in range(S_min,S_max+1):
#            Yhat = Yhat + ((L==s)*A) @ np.roll(X,s,axis=1)
#       
#        uA = np.zeros((I,J))
#        for s in range(S_min,S_max+1):
#            uA = uA + (L==s) * (Y @ np.transpose(np.roll(X,s))) / ((Yhat @ np.transpose(np.roll(X,s))) + alpha_A)
#        A = np.clip(A * uA,0,A_max)
#
##        grad_L_D = np.zeros((I,J))
##        for s in range(S_min,S_max+1):
##            grad_L_D = grad_L_D + (L==s) * ((Y-Yhat) @ np.transpose(np.roll(dX,s)))
##        grad_L_D = -A * grad_L_D    
##
##        L = np.clip(L - np.sign(grad_L_D) * np.heaviside(np.fabs(grad_L_D)-10,1),S_min,S_max)
#
#        delta_Lp1_D_overA = np.zeros((I,J))
#        delta_Lm1_D_overA = np.zeros((I,J))
#        for s in range(S_min,S_max+1):
#            delta_Lp1_D_overA = delta_Lp1_D_overA + (L==s) * ((Y-Yhat) @ np.roll(dp1X,s).T)
#            delta_Lm1_D_overA = delta_Lm1_D_overA + (L==s) * ((Y-Yhat) @ np.roll(dm1X,s).T)
#        delta_Lp1_D_overA = -delta_Lp1_D_overA + 0.5*A*M     
#        delta_Lm1_D_overA = -delta_Lm1_D_overA + 0.5*A*M     
#        
#        uL =  np.sign(delta_Lm1_D_overA*(delta_Lm1_D_overA<0) - delta_Lp1_D_overA*(delta_Lp1_D_overA<0))
#        L = np.clip(L + uL,S_min,S_max)
#        
#        if count%10 == 0: 
#            norm_uA = np.linalg.norm((uA - 1)*A)
#            norm_uL = np.linalg.norm((uL)*(L>S_min)*(L<S_max))
#            #print('norm_uA=%f  norm_uL=%f   at %i iteration' % (norm_uA,norm_uL,count) )
#            if (norm_uA < convergence_threshold or abs(norm_uA/prev_norm_uA-1) < convergence_threshold) \
#                and (norm_uL < convergence_threshold or abs(norm_uL/prev_norm_uL-1) < convergence_threshold):
#                converged = True
#                print('Converged at %i iterations' % count)
#            prev_norm_uA=norm_uA
#            prev_norm_uL=norm_uL
#            
#        count = count+1
#        
#    A[A<A_min] = 0
#    return (A,L,Yhat)     
#        
#S_min=-8
#S_max=2
#
#i=1
#
#i=np.random.randint(0,I)
#plt.plot(Yhat[i,:])
#plt.plot(Y[i,:])
#print(A[i,:])
#print(L[i,:])
#
#A.shape
#
#plt.gcf().set_size_inches(4,40)
#plt.imshow(A, cmap='gray', interpolation='nearest')
#plt.show()
#
#plt.plot(np.sum(A,0))
#
#np.sum(A,0) > 50
#
#X1 = X[np.sum(A,0) > 50,:]
#
#X=X1
#
#G.subset(np.sum(A,0) > 50).plot()
#
#G.plot()
#
#
#grad_L_D[i,:]
#
#alpha_A=100
#alpha_X=0
#
#A[i,:]
#L[i,:]
#
#X = G.array
#G.plot()
#
#plt.hist(A.flatten(),100)
#
#plt.hist(grad_L_D.flatten(),100)
#
#Spe.Ens(X,y=None,times=X_times).plot()
#Spe.Ens(dX,y=None,times=X_times).plot()


































#
#
#
#def NMF_ISRA(Y, A=None, X=None, J=None, maxIter=1000, alpha_A=0, alpha_X=0)
#    """
#    ISRA algorithm for Nonnegative Matrix Factorization (NMF)
#            
#    INPUT
#    Y - Matrix of I x T representing data 
#    A - Matrix of I x J representing weights of each gesture in each instance
#    X - Matrix of J x T representing each gesture at each time
#    
#    J - Rank of the decomposition (number of gestures to use)
#        Defaults to 10 if X not given
#    maxIter - max number of iterations.
#    alpha_A - sparsness parameter for A.     
#    alpha_X - sparsness parameter for X.     
#       
#    OUTPUT
#    A and X such that ||Y − A ∗ X|| is minimized
#    
#    This function was inspired by code by Anh Huy Phan anh Andrzej Cichocki 2008
#    """
#
#    eps = np.finfo(float).eps
#
#    # Forcing Y to positive matrix
#    Y = np.nan_to_num(Y)
#    Y[Y <= 0] = eps
#    (I,T) = np.shape(Y)
#
#    #checkign or initializing inputs
#    if X is None:
#        if J is not None:
#            X = np.random.rand(J,T)
#            X[X<=0] = eps
#        else:
#             raise Error("J not defined")
#    else:
#        X = np.nan_to_num(X)
#        X[X<=0] = eps
#        (J1,T1) = np.shape(X)
#        if J is not None and J != J1:
#            raise Error("J inconsistent with X")
#        if T != T1:
#            raise Error("dimension of X inconsistent with that of Y")
#        J=J1
# 
#    if A is None:
#        A = np.random.rand(I,J)
#        A[A<=0] = eps
#    else:
#        A = np.nan_to_num(A)
#        A[A<0] = eps
#        (I2,J2) = np.shape(A)
#        if J != J2:
#            raise Error("A inconsistent with J or X")
#        if I != T2:
#            raise Error("dimension of A inconsistent with that of Y")
#       
#    for k in range(1,maxIter):
#        Yhat = A @ X
#        
#        A = np.clip(A * (Y @ np.transpose(X)) / ((Yhat @ np.transpose(X)) + alpha_A + eps),eps,2)
#        
#        #X = np.clip(X * (np.transpose(A) @ Y) / ((np.transpose(A) @ Yhat) + alpha_X + eps),eps,16)        
#        
#        #Spe.Ens(X,y=None,times=ens.times).plot()
#
