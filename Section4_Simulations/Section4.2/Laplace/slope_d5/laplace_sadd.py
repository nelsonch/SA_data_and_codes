#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import math
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from saddlepoints import saddle_points_app


class laplace_sda(saddle_points_app):
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
        tg = 1/(1 - self.b_v*self.b_v*self.Z*self.Z*xi[:, np.newaxis]*xi[:, np.newaxis])
        #tg = self.b_v /(self.b_v - self.Z*xi[:, np.newaxis])
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
        hh1 = 2*self.Z*self.Z*self.Z*self.Z*self.b_v*self.b_v*self.b_v*self.b_v*xi[:, np.newaxis]*xi[:, np.newaxis] + 2*self.Z*self.Z*self.b_v*self.b_v     
        hh2 = (1 - self.b_v*self.b_v*self.Z*self.Z*xi[:, np.newaxis]*xi[:, np.newaxis])**2
        tg = hh1/hh2
        #tg = (self.Z*self.Z/(self.b_v - self.Z*xi[:, np.newaxis])/(self.b_v - self.Z*xi[:, np.newaxis]))
        tg1 = np.where(np.isnan(tg)|(tg==np.inf)|(tg==-np.inf), 0, tg)
        #tg2 = np.nan_to_num(tg)
        tem22 = (tg1@self.m_vector).ravel()
        #tem2 = np.where(np.isnan(tem22)|(tem22==np.inf)|(tem22==-np.inf), 0, tem22)
        final = tem22 + self.sigma*self.sigma
        return final

    
 

######### for the equlity constraints
class equlity_constraints_laplace():
    def __init__(self, trust_tol=1e-6, RF_index=None, X = None, Y = None, Z = None):
        self.X = X 
        self.Y = Y
        self.Z = Z 
        self.N, self.n = X.shape
        self.d = self.n 
        self.trust_tol = trust_tol
        self.RF_true = np.where(RF_index == 'Y')[0].astype(int)
        self.m_vector = np.ones(len(self.RF_true))[:, np.newaxis]
        #self.ss = int(X.shape[0]/20) #### g=20
    
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

    def gradient_K_fun_laplace(self, x):
        xi = self._get_parameters(x)
        #self.xi = xi
        tem1 = (self.X@self.beta_v[:, np.newaxis]).ravel()
        tem3 = (self.sigma**2)*(xi)
        ##### term2 
        tem4 = 1 - self.b_v*self.b_v*self.Z*self.Z*xi[:, np.newaxis]*xi[:, np.newaxis]
        tg = (2*self.b_v*self.b_v*self.Z*self.Z*xi[:, np.newaxis])/tem4
        #tg = self.Z/(self.b_v - self.Z*xi[:, np.newaxis])
        tg1 = np.where(np.isnan(tg)|(tg==np.inf)|(tg==-np.inf), 0, tg)
        tem2 = (tg1@self.m_vector).ravel()
        final = tem1 + tem2 + tem3
        return final
             

    def ineq_constraints_laplace(self, x):
        xi = self._get_parameters(x)
        tem = self.b_v*self.b_v*self.Z*self.Z*xi[:, np.newaxis]*xi[:, np.newaxis]
        #tem = self.b_v - self.Z*xi[:, np.newaxis]
        final = tem.ravel(order='F')
        return final


class step2_RF():
    def __init__(self, beta, sigma, b_v, g, X, Y, Z_long):
        self.beta = beta
        self.sigma = sigma
        self.b_v = b_v
        self.g = g
        self.X = X 
        self.Y = Y 
        self.Z_long = Z_long
        self.ss = int(X.shape[0]/g)
        self.m_vector = np.array([1, 1])[:, np.newaxis]

    def RF_step2_laplace(self, x):
        v = x[0:self.g]
        gamma = x[self.g:2*self.g]
        E = self.Y.ravel() - (self.X@self.beta[:, np.newaxis]).ravel() - (self.Z_long@gamma[:, np.newaxis]).ravel()
        tem1 = np.sum(E**2/self.sigma**2)
        ##### term2
        #gamma_m = gamma.reshape(g, k, order='F')
        tem2 = 2*self.ss*np.sum(v/self.b_v)
        final = tem1 + tem2 
        return final

    def step2_constraints(self, x):
        v = x[0:self.g]; gamma=x[self.g:2*self.g];
        final = np.concatenate((v-gamma, v+gamma))
        return final 


    def RF_step2_laplace_k2(self, x):
        ###### k=1 
        v1 = x[0:self.g]
        gamma1 = x[self.g:2*self.g]
        v2 = x[2*self.g:3*self.g]
        gamma2 = x[3*self.g:4*self.g]
        ##########
        gamma = np.concatenate((gamma1, gamma2))[:, np.newaxis]
        v = np.concatenate((v1[:, np.newaxis], v2[:, np.newaxis]), 1)
        ###########    
        E = self.Y.ravel() - (self.X@self.beta[:, np.newaxis]).ravel() - (self.Z_long@gamma).ravel()
        tem1 = np.sum(E**2/self.sigma**2)
        ##### term2
        #gamma_m = gamma.reshape(g, k, order='F')
        tg = v/self.b_v[np.newaxis, :]
        tem2 = 2*self.ss*np.sum((tg@self.m_vector).ravel())
        final = tem1 + tem2 
        return final

    def step2_constraints_k2(self, x):
        v1 = x[0:self.g]
        gamma1=x[self.g:2*self.g]
        v2 = x[2*self.g:3*self.g] 
        gamma2=x[3*self.g:4*self.g]
        #v = x[0:self.g]; gamma=x[self.g:2*self.g];
        final = np.concatenate((v1-gamma1, v1+gamma1, v2-gamma2, v2+gamma2))
        return final

def RF_step2_laplace_simple(gamma, beta, sigma, g, X, Y, Z_long):
        E = Y.ravel() - (X@beta[:, np.newaxis]).ravel() - (Z_long@gamma).ravel()
        tem1 = np.sum(E**2/sigma**2)
        return tem1

def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)