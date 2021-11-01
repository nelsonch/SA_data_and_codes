#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import math
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from saddlepoints import saddle_points_app


class triangular_sda(saddle_points_app):
    def _get_parameters(self, x):
        self.beta_v = np.abs(x[0:self.n])
        self.sigma = x[self.n:(self.n+1)]
        xi = x[(self.n+1):]
        self.beta = self.beta_v[self.RF_true]
        return xi


    def _K_fun(self, xi):
        tem1 = ((self.X@self.beta_v[:, np.newaxis]).ravel())*xi
        tem3 = (xi**2)*(self.sigma**2)/2
        #####
        tem = np.exp(self.beta[np.newaxis, :]*self.Z*xi[:, np.newaxis])
        num = tem + 1/tem -2
        num1 = np.where(num==0, 1, num)
        num2 = np.nan_to_num(num1)
        ####
        den = self.beta[np.newaxis, :]*self.beta[np.newaxis, :]*self.Z*self.Z*xi[:, np.newaxis]*xi[:, np.newaxis]
        den1 = np.where(den==0, 1, den)
        den2 = np.nan_to_num(den1)
        tg = np.log(num2/den2)
        tem22 = (tg@self.m_vector).ravel()
        tem2 = np.where(np.isnan(tem22)|(tem22==np.inf)|(tem22==-np.inf), 0, tem22)
        final = np.sum(tem1+tem3)+np.sum(tem2)
        return final 

    
    def _gradient2_K_fun(self, xi):
        tem = np.exp(self.beta[np.newaxis, :]*self.Z*xi[:, np.newaxis])
        tem2 = tem*tem
        num = 2*tem2 + 2 - 2*tem*(self.beta[np.newaxis, :]*self.beta[np.newaxis, :]*xi[:, np.newaxis]*xi[:, np.newaxis]*self.Z*self.Z + 2)
        #num1 = np.where(num==0, 0, num)
        num2 = np.nan_to_num(num)
        ##### den
        den = xi[:, np.newaxis]*xi[:, np.newaxis]*(tem - 1)*(tem-1)
        den1 = np.where(den==0, 1, den)
        den2 = np.nan_to_num(den1)
        ####
        tg = np.log(num2/den2)
        tem22 = (tg@self.m_vector).ravel()
        tem2 = np.where(np.isnan(tem22)|(tem22==np.inf)|(tem22==-np.inf), 0, tem22) 
        final = tem2 + self.sigma*self.sigma
        return final    
    

######### for the equlity constraints
class equlity_constraints_triangular():
    def __init__(self, trust_tol=1e-6, RF_index=None, X = None, Y = None, Z = None):
        self.X = X 
        self.Y = Y
        self.Z = Z 
        self.N, self.n = X.shape
        self.d = self.n 
        self.trust_tol = trust_tol
        self.RF_true = np.where(RF_index == 'Y')[0].astype(int)
        self.m_vector = np.ones(len(self.RF_true))[:, np.newaxis]

    def gradient_K_fun_triangular(self, x):
        beta_v = x[0:self.d]
        sigma = x[self.d:(self.d+1)]
        xi = x[(self.d+1):]
        tem1 = (self.X@beta_v[:, np.newaxis]).ravel()
        tem3 = (sigma**2)*(xi)
        beta = beta_v[self.RF_true]
        ######
        tem = np.exp(beta[np.newaxis, :]*self.Z*xi[:, np.newaxis])
        num = tem*(beta[np.newaxis, :]*self.Z*xi[:, np.newaxis] - 2) + beta[np.newaxis, :]*self.Z*xi[:, np.newaxis] + 2
        #num1 = np.where(num==0, 0, num)
        num2 = np.nan_to_num(num)
        ######
        den = xi[:, np.newaxis]*(tem - 1)
        den1 = np.where(den==0, 1, den)
        den2 = np.nan_to_num(den1)
        ######
        tem2 = num2/den2
        #tem4 = np.where(np.isnan(tem2), 0, tem2)
        tem22 = (tem2@self.m_vector).ravel()
        tem8 = np.where(np.isnan(tem22)|(tem22==np.inf)|(tem22==-np.inf), 0, tem22) 
        final = tem1 + tem8 + tem3
        return final

    

class step2_RF():
    def __init__(self, beta, sigma, g, X, Y, Z_long):
        self.beta = beta
        self.sigma = sigma
        self.g = g
        self.X = X 
        self.Y = Y 
        self.Z_long = Z_long
        self.ss = int(X.shape[0]/g)
        self.m_vector = np.array([1, 1])[:, np.newaxis]

    def RF_step2_triangular(self, x):
        ###### k=1
        #gamma = x 
        v = x[0:self.g]; gamma=x[self.g:2*self.g];
        E = self.Y.ravel() - (self.X@self.beta[:, np.newaxis]).ravel() - (self.Z_long@gamma[:, np.newaxis]).ravel()
        tem1 = np.sum(E**2/self.sigma**2)
        #######
        tem = np.log((self.beta[0] - v)/self.beta[0]**2)
        tg1 = np.where(np.isnan(tem)|(tem==np.inf)|(tem==-np.inf), 0, tem)
        tem2 = -2*self.ss*np.sum(tg1)
        #gamma_m = gamma.reshape(g, k, order='F')
        #tem = (a_v[np.newaxis, :] - 1)*np.log(gamma_m) - b_v[np.newaxis, :]*gamma_m
        #tem2 = -2*g*np.sum((tg1@m_vector).ravel())
        final = tem1 + tem2
        return final 

    def step2_constraints(self, x):
        v = x[0:self.g]; gamma=x[self.g:2*self.g];
        final = np.concatenate((v-gamma, v+gamma))
        return final 


    def RF_step2_triangular_k2(self, x):
        ###### k=1 
        v1 = x[0:self.g] 
        gamma1 = x[self.g:2*self.g]
        v2 = x[2*self.g:3*self.g]
        gamma2 = x[3*self.g:4*self.g]

        ######
        gamma = np.concatenate((gamma1, gamma2))[:, np.newaxis]
        v = np.concatenate((v1[:, np.newaxis], v2[:, np.newaxis]), 1)
        #######
        E = self.Y.ravel() - (self.X@self.beta[:, np.newaxis]).ravel() - (self.Z_long@gamma).ravel()
        tem1 = np.sum(E**2/self.sigma**2)
        #######
        tem = np.log((self.beta[0:2][np.newaxis, :] - v)/(self.beta[0:2][np.newaxis, :]**2))
        tg1 = np.where(np.isnan(tem)|(tem==np.inf)|(tem==-np.inf), 0, tem)
        tem2 = -2*self.ss*np.sum((tg1@self.m_vector).ravel())
        final = tem1 + tem2
        return final

    def RF_step2_triangular_k2_noabs(self, x):
        ###### k=1 
        gamma1 = x[0:self.g] 
        gamma2 = x[self.g:2*self.g]
        v1 = np.abs(gamma1)
        v2 = np.abs(gamma2)
        #v2 = x[2*self.g:3*self.g]
        #gamma2 = x[3*self.g:4*self.g]
        ######
        gamma = np.concatenate((gamma1, gamma2))[:, np.newaxis]
        v = np.concatenate((v1[:, np.newaxis], v2[:, np.newaxis]), 1)
        #######
        E = self.Y.ravel() - (self.X@self.beta[:, np.newaxis]).ravel() - (self.Z_long@gamma).ravel()
        tem1 = np.sum(E**2/self.sigma**2)
        #######
        tem = np.log((self.beta[0:2][np.newaxis, :] - v)/(self.beta[0:2][np.newaxis, :]**2))
        tg1 = np.where(np.isnan(tem)|(tem==np.inf)|(tem==-np.inf), 0, tem)
        tem2 = -2*self.ss*np.sum((tg1@self.m_vector).ravel())
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
    

def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


def get_dummy_own(x, g=20):
    N = len(x)
    ss = int(N/g)
    tem = np.zeros((N, g))
    for i in range(g):
        tem[ss*i:ss*(i+1), i] = x[ss*i:ss*(i+1)]
    return tem