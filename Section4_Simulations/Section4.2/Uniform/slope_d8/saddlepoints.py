#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import math
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

class saddle_points_app():
    def __init__(self, trust_tol=1e-6, g=None, RF_index=None, X=None, Y=None, Z=None):
        self.X = X 
        self.Y = Y
        self.Z = Z 
        self.N, self.n = X.shape
        self.d = self.n
        self.g = g 
        self.trust_tol = trust_tol
        self.RF_true = np.where(RF_index == 'Y')[0].astype(int)
        self.m_vector = np.ones(len(self.RF_true))[:, np.newaxis]

    def LK_negative(self, x):
        xi = self._get_parameters(x)
        tg = self._gradient2_K_fun(xi)
        tg1 = np.where(np.isnan(tg), -200, tg)
        tem1 = np.sum(0.5*np.log(tg1[(tg1>0)&(tg1<np.inf)]))
        tem2 = -1*self._K_fun(xi)
        tem3 = np.sum(xi*self.Y)
        term1 = tem1 + tem3 + tem2
        return term1 

    @abstractmethod
    def _get_parameters(self, x):
        pass

    @abstractmethod
    def _K_fun(self, xi):
        pass

    @abstractmethod
    def _gradient2_K_fun(self, xi):
        pass    