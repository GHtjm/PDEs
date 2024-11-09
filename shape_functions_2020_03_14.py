# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:18:41 2020

@author: dumba
"""
import numpy as np

""" Define the shape functions \phi (x) """

def phi_j(x,xm,j,df):
    idx = len(x)
    phi = np.zeros(x.shape)
    
    if j==0: #Special case: \phi_0 (x)
        for i in range(0,idx):
            if x[i] < xm[j+1]:
                phi[i] = (xm[j+1] - x[i])/(xm[j+1] - xm[j])
                
    elif j==df: #Special case: \phi_M (x)
        for i in range(0,idx):
            if x[i] > xm[j-1]:
                phi[i] = (x[i] - xm[j-1])/(xm[j] - xm[j-1])
                
    else: #Remaining cases \phi_j (x), j=1,...,M-1
        for i in range(0,idx):
            if x[i] < xm[j-1]:
                phi[i] = 0
            elif x[i] > xm[j+1]:
                phi[i] = 0
            elif x[i] < xm[j]:
                phi[i] = (x[i] - xm[j-1])/(xm[j] - xm[j-1])
            else:
                phi[i] = (xm[j+1] - x[i])/(xm[j+1] - xm[j])
        
    return phi
        

""" Define the derivatives of shape function \phi(x) """

def dphi_j(x,xm,j,df):
    idx = len(x)
    dphi = np.zeros(x.shape)
    
    if j == 0: #Special case: \phi_0 (x)
        for i in range(0,idx):
            if x[i] < xm[j+1]:
                dphi[i] = -1/(xm[j+1] - xm[j])
    
    elif j == df:  #Special case: \phi_M (x)
        for i in range(0,idx):
            if x[i] > xm[j-1]:
                dphi[i] = 1/(xm[j] - xm[j-1])
    
    else: #Remaining cases \phi_j (x), j=1,...,M-1
        for i in range(0,idx):
            if x[i] < xm[j-1]:
                dphi[i] = 0
            elif x[i] > xm[j+1]:
                dphi[i] = 0
            elif x[i] < xm[j]:
                dphi[i] = 1/(xm[j] - xm[j-1])
            else:
                dphi[i] = -1/(xm[j+1] - xm[j])
        
    return dphi

"""

    Define the integral function: (f,g) = \int_0^1 f(x) g(x)  dx

    As the shape functions consist of straight lines, we simply use the trapezium rule.

"""

def int_trap(x,f,g,hh): #Trapezium rule for f(x) * g(x)
    fg = f * g
    return hh * (sum(fg) - (fg[0]+fg[-1])/2)


"""

    Solve the lienar algebra problem:
        K u = F
    Taking advantage of the knowledge that K is a banded matrix
    
"""
import scipy.linalg as la #Linear algebra solver

def Banded_solve(StiffnessM,ForceV,df):
    
    """ Determine the banding for K.
        See for details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html
        In our case we expect to find ii=jj=1."""
    jj = 0
    for j in range(1,df):
        if StiffnessM[0,j] != 0:
            jj+= 1
    
    ii = 0
    for i in range(1,df):
        if StiffnessM[i,0] != 0:
            ii+=1
   
    if ii == df - 1 or jj == df - 1:
        """ Matrix not banded, use conventional solver """
        if df >= 3: #Matrix not expected to be banded for df =1 or df=2
            import warnings
            warnings.warn('STIFFNESS MATRIX IS NOT BANDED')
            warnings.warn('USING CONVENTIONAL SOLVER')
        
        ui = la.solve(StiffnessM,ForceV) #Solve the linear algebra problem: K u = F   
    else:
        """ Here the matrix ab is simply a more efficient way of representing K 
            We remove most of the zeros, leaving a matrix with dimensions: (ii+jj+1) X df 
            Computations on this new matrix are performed using the banded solver.
            
            See for details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html """
            
        ab = np.zeros(np.linspace(0,1,df).shape)
        abitt = np.zeros(np.linspace(0,1,df).shape)
        
        for j in range(0,ii+jj):
            ab = np.vstack([ab,abitt])
                
        
        for i in range(0,df):
            for j in range(0,df):        
                if ii+i-j>=0 and ii+i-j <= 2:           
                    ab[ii + i - j, j] = StiffnessM[i,j]
            
                
        ui = la.solve_banded((jj,ii),ab,ForceV) #Solve the linear algebra problem: K u = F noting that K is banded
    
    return ui