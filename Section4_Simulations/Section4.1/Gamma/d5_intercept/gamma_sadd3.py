#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import math
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from saddlepoints import saddle_points_app


class gamma_sda(saddle_points_app):
    def _get_parameters(self, x):
        self.beta_v = np.abs(x[0:self.n])
        k = int(len(self.RF_true))
        self.sigma = x[self.n: (self.n+1)] 
        self.b_v = x[(self.n+1): (self.n+1+k)][np.newaxis, :] 
        #self.b_v = x[(self.n+1+k):(self.n+1+2*k)][np.newaxis, :]
        #self.sigma = x[(self.n+2*k):(self.n+2*k+1)]
        xi = x[(self.n+1+k):]
        self.beta = self.beta_v[self.RF_true]
        return xi


    def _K_fun(self, xi):
        tem1 = ((self.X@self.beta_v[:, np.newaxis]).ravel())*xi
        tem3 = (xi**2)*(self.sigma**2)/2
        ###### term 2
        tg = self.b_v /(self.b_v - self.Z*xi[:, np.newaxis])
        tg1 = np.where(tg<=0, 1, tg)
        tg2 = np.nan_to_num(tg1)
        gg = np.log(tg2)
        tem21 = np.where(np.isnan(gg)|(gg==np.inf)|(gg==-np.inf), 0, gg)
        #tem21 = np.where(np.isnan(gg), 0, gg)
        tem22 = (tem21@self.m_vector).ravel()
        #tem2 = np.where(np.isnan(tem22)|(tem22==np.inf)|(tem22==-np.inf), 0, tem22)
        final = np.sum(tem1+tem3) + np.sum(tem22)
        return final 

    
    def _gradient2_K_fun(self, xi):
        tg = (self.Z*self.Z/(self.b_v - self.Z*xi[:, np.newaxis])/(self.b_v - self.Z*xi[:, np.newaxis]))
        tg1 = np.where(np.isnan(tg)|(tg==np.inf)|(tg==-np.inf), 0, tg)
        #tg2 = np.nan_to_num(tg)
        tem22 = (tg1@self.m_vector).ravel()
        #tem2 = np.where(np.isnan(tem22)|(tem22==np.inf)|(tem22==-np.inf), 0, tem22)
        final = tem22 + self.sigma*self.sigma
        return final

    
 

######### for the equlity constraints
class equlity_constraints_gamma():
    def __init__(self, trust_tol=1e-6, RF_index=None, X = None, Y = None, Z = None):
        self.X = X 
        self.Y = Y
        self.Z = Z 
        self.N, self.n = X.shape
        self.d = self.n 
        self.trust_tol = trust_tol
        self.RF_true = np.where(RF_index == 'Y')[0].astype(int)
        self.m_vector = np.ones(len(self.RF_true))[:, np.newaxis]

    
    def _get_parameters(self, x):
        self.beta_v = np.abs(x[0:self.n])
        k = int(len(self.RF_true))
        self.k = k
        self.sigma = x[self.n: (self.n+1)] 
        self.b_v = x[(self.n+1): (self.n+1+k)][np.newaxis, :] 
        #self.b_v = x[(self.n+1+k):(self.n+1+2*k)][np.newaxis, :]
        #self.sigma = x[(self.n+2*k):(self.n+2*k+1)]
        xi = x[(self.n+1+k):]
        self.beta = self.beta_v[self.RF_true]
        return xi

    def gradient_K_fun_gamma(self, x):
        xi = self._get_parameters(x)
        #self.xi = xi
        tem1 = (self.X@self.beta_v[:, np.newaxis]).ravel()
        tem3 = (self.sigma**2)*(xi)
        ##### term2 
        tg = self.Z/(self.b_v - self.Z*xi[:, np.newaxis])
        tg1 = np.where(np.isnan(tg)|(tg==np.inf)|(tg==-np.inf), 0, tg)
        tem2 = (tg1@self.m_vector).ravel()
        final = tem1 + tem2 + tem3
        return final

    def gradient2_K_fun_gamma(self, x):
        xi = self._get_parameters(x)
        tg = (self.Z*self.Z/(self.b_v - self.Z*xi[:, np.newaxis])/(self.b_v - self.Z*xi[:, np.newaxis]))
        tg1 = np.where(np.isnan(tg)|(tg==np.inf)|(tg==-np.inf), 0, tg)
        #tg2 = np.nan_to_num(tg)
        tem22 = (tg1@self.m_vector).ravel()
        #tem2 = np.where(np.isnan(tem22)|(tem22==np.inf)|(tem22==-np.inf), 0, tem22)
        final = tem22 + self.sigma*self.sigma
        ########## 
        k1 = self.X[:, 0:self.d]
        k2 = np.array(2*xi)[:, np.newaxis]
        hh = (self.b_v - self.Z*xi[:, np.newaxis])
        k3 = self.Z/hh 
        k4 = -1*self.Z/(hh*hh)
        aa = np.concatenate((k1, k2, k3, k4), 1)
        #aa = np.zeros((self.N, self.d+1+self.k*2))
        #bb = np.zeros((d+1, N))
        #tem1 = np.concatenate((bb, np.diag(final)), 0)
        final_f = np.concatenate((aa, np.diag(final)), 1)
        return final_f
             

    def ineq_constraints_gamma(self, x):
        xi = self._get_parameters(x)
        tem = self.b_v - self.Z*xi[:, np.newaxis]
        final = tem.ravel(order='F')
        return final


def RF_step2_gamma(gamma, beta, sigma, a_v, b_v, k, g, m_vector, X, Y, Z_long):
    E = Y.ravel() - (X@beta[:, np.newaxis]).ravel() - (Z_long@gamma).ravel()
    tem1 = np.sum(E**2/sigma**2)
    ##### term2
    gamma_m = gamma.reshape(g, k, order='F')
    tem = (a_v[np.newaxis, :] - 1)*np.log(gamma_m) - b_v[np.newaxis, :]*gamma_m
    tg1 = np.where(np.isnan(tem)|(tem==np.inf)|(tem==-np.inf), 0, tem)
    tem2 = -2*g*np.sum((tg1@m_vector).ravel())
    final = tem1 + tem2 
    return final

def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)