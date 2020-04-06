#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMF.py module of WarblePy.

Provides functions to perform lagged Non-Negative Matrix Factorization

Author: Alan Bush
"""

import numpy as np

class Error(Exception): pass

def NMF_ISRA_lag(Y, A=None, X=None, L=None, S=0, J=None, \
                 fit_A=None, fit_X=None, fit_L=None, \
                 A_max=None, A_min=None, A_update_factor_lim=[0.7071,1.4142],
                 X_max=None, X_min=None, X_update_factor_lim=[0.7071,1.4142],
                 alpha_A=0, alpha_X=0, alpha_L=0,  \
                 maxIter=1000, convergence_threshold=1e-3):
    """
    Nonnegative Matrix Factorization (NMF) with lags.
    Implemented using Lee-Seung type Image Space Reconstruction Algorithm (ISRA)

    INPUT
    Y - Matrix of I x T representing data
    A - Matrix of I x J representing weights of each gesture in each instance
    X - Matrix of J x T representing each gesture at each time
    L - Integer matrix of I x J repesenting 'lags' of each component in number of samples

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
    alpha_X - sparsness parameter for X.
    alpha_L - smoothness parameter for L.

    maxIter - max number of iterations.
    convergence_threshold - fractional change of convergence between successive iterations

    OUTPUT
    A, X and L such that

        D = 0.5*||Y − Yhat||_2^2 + alpha_A*||A||_1 + alpha_X*||X||_1

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

    dp1X = roll0(X,+1)-X
    dm1X = roll0(X,-1)-X
    IJx1 = np.ones((I,J))
    M = IJx1 @ ((dp1X @ dp1X.T) * np.identity(J))
    G = 2 * np.identity(J) - roll0(np.identity(J),+1) - roll0(np.identity(J),-1)
    G[0,0]=1
    G[J-1,J-1]=1
    H = np.ones((I,J))
    H[:,0]=1/2
    H[:,J-1]=1/2

    count = 0
    converged = False
    Dlast = np.inf

    while count < maxIter and not converged:

        #updating Yhat
        Yhat = np.zeros((I,T))
        for s in range(S_min,S_max+1):
            Yhat = Yhat + ((L==s)*A) @ roll0(X,s)
        D = 0.5*np.linalg.norm(Y-Yhat)**2

        #updating A
        if fit_A:
            #uA = np.zeros((I,J))
            uAnum = np.zeros((I,J))
            uAdenom = np.zeros((I,J))
            for s in range(S_min,S_max+1):
                uAnum = uAnum + (L==s) * (Y @ roll0(X,s).T)
                uAdenom = uAdenom + (L==s) * (Yhat @ roll0(X,s).T)
            uA = uAnum / (uAdenom + alpha_A + eps)
            uA = np.clip(uA, A_update_factor_lim[0], A_update_factor_lim[1])
            A = np.clip(A * uA,0,A_max)

        #updating X
        if fit_X:
            uXnum = np.zeros((J,T))
            uXdenom = np.zeros((J,T))
            for s in range(S_min,S_max+1):
                uXnum = uXnum + ((L==s).T * A.T) @ roll0(Y,-s)
                uXdenom = uXdenom + ((L==s).T * A.T) @ roll0(Yhat,-s)
            uX = uXnum / (uXdenom + alpha_X + eps)
            uX = np.clip(uX, X_update_factor_lim[0], X_update_factor_lim[1])
            X = np.clip(X * uX,0,X_max)

        # updating L
        if fit_L:
            delta_Lp1_D_overA = np.zeros((I,J))
            delta_Lm1_D_overA = np.zeros((I,J))
            for s in range(S_min,S_max+1):
                delta_Lp1_D_overA = delta_Lp1_D_overA + (L==s) * ((Y-Yhat) @ roll0(dp1X,s).T)
                delta_Lm1_D_overA = delta_Lm1_D_overA + (L==s) * ((Y-Yhat) @ roll0(dm1X,s).T)
            delta_Lp1_D = -A*delta_Lp1_D_overA + 0.5*A*A*M + alpha_L*(H + L @ G)
            delta_Lm1_D = -A*delta_Lm1_D_overA + 0.5*A*A*M + alpha_L*(H - L @ G)
            uL =  np.sign(delta_Lm1_D*(delta_Lm1_D<0) - delta_Lp1_D*(delta_Lp1_D<0))
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



def roll0(a, shift):
    """
    Permute elements of matix a shifting them 'shift' places to the right.
    Rightmost elements are discarded and the leftmost columns asigned to zero.
    Use negative shifts to displace to the left.

    INPUT
    a - array_like, Input array.

    shift - int, The number of places by which elements are shifted.

    OUTPUT
    res - ndarray, Output array, with the same shape as a.
    """
    if shift==0:
        return(a)

    res = np.roll(a,shift,axis=1)
    if shift>0:
        res[:,:shift] = 0
    else:
        res[:,shift:] = 0

    return(res)
