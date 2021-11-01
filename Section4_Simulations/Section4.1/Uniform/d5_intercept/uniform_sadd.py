#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import math
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from saddlepoints import saddle_points_app


class uniform_sda(saddle_points_app):
    def _get_parameters(self, x):
        self.beta_v = np.abs(x[0:self.n])
        self.sigma = x[self.n:(self.n+1)]
        xi = x[(self.n+1):]
        self.beta = self.beta_v[self.RF_true]
        return xi


    def _K_fun(self, xi):
        tem1 = ((self.X@self.beta_v[:, np.newaxis]).ravel())*xi
        tem3 = (xi**2)*(self.sigma**2)/2
        tem = np.exp(self.beta[np.newaxis, :]*self.Z*xi[:, np.newaxis])
        num = tem - 1/tem 
        num1 = np.where(num==0, 1, num)
        num2 = np.nan_to_num(num1)

        den = 2*self.beta[np.newaxis, :]*self.Z*xi[:, np.newaxis]
        den1 = np.where(den==0, 1, den)
        den2 = np.nan_to_num(den1)
        #tg1 = (tem - 1/tem)/num
        #tg2 = np.where(np.isnan(tg1), 1, tg1)
        tg = np.log(num2/den2)
        tem22 = (tg@self.m_vector).ravel()
        tem2 = np.where(np.isnan(tem22)|(tem22==np.inf)|(tem22==-np.inf), 0, tem22)
        final = np.sum(tem1+tem3)+np.sum(tem2) 
        return final 

    
    def _gradient2_K_fun(self, xi):
        tem = np.exp(2*self.beta[np.newaxis, :]*self.Z*xi[:, np.newaxis])
        den = xi[:, np.newaxis]*xi[:, np.newaxis]*(tem - 1)*(tem - 1)
        den1 = np.where(den==0, 1, den) 
        den2 = np.nan_to_num(den1)
        num = 1 + tem*tem - 2*tem*(2*self.beta[np.newaxis, :]*self.beta[np.newaxis, :]*self.Z*self.Z*xi[:, np.newaxis]*xi[:, np.newaxis] + 1)
        num2 = np.nan_to_num(num)
        #num1 = np.where(num==0, 0, num)
        tem2 = num2/den2
        #tem3 = np.where(np.isnan(tem2), 0, tem2)
        tem22 = (tem2@self.m_vector).ravel()     
        final = tem22 + self.sigma*self.sigma
        return final    
    
 

######### for the equlity constraints
class equlity_constraints_uniform():
    def __init__(self, trust_tol=1e-6, RF_index=None, X = None, Y = None, Z = None):
        self.X = X 
        self.Y = Y
        self.Z = Z 
        self.N, self.n = X.shape
        self.d = self.n 
        self.trust_tol = trust_tol
        self.RF_true = np.where(RF_index == 'Y')[0].astype(int)
        self.m_vector = np.ones(len(self.RF_true))[:, np.newaxis]

    def gradient_K_fun_uniform(self, x):
        beta_v = x[0:self.d]
        sigma = x[self.d:(self.d+1)]
        xi = x[(self.d+1):]
        tem1 = (self.X@beta_v[:, np.newaxis]).ravel()
        tem3 = (sigma**2)*(xi)

        beta = beta_v[self.RF_true]
        tem = np.exp(2*beta[np.newaxis, :]*self.Z*xi[:, np.newaxis])
        den = xi[:, np.newaxis]*(tem - 1)
        den1 = np.where(den==0, 1, den)
        den2 = np.nan_to_num(den1)
        num = 1 + tem*(beta[np.newaxis, :]*self.Z*xi[:, np.newaxis] - 1) + beta[np.newaxis, :]*self.Z*xi[:, np.newaxis]
        num2 = np.nan_to_num(num)
        tem2 = num2/den2
        #tem4 = np.where(np.isnan(tem2), 0, tem2)
        tem22 = (tem2@self.m_vector).ravel()
        final = tem1 + tem22 + tem3
        return final
        

def RF_step2_uniform(gamma, beta, sigma, X, Y, Z_long):
    E = Y.ravel() - (X@beta[:, np.newaxis]).ravel() - (Z_long@gamma).ravel()
    tem1 = np.sum(E**2/sigma**2)
    return tem1


def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)