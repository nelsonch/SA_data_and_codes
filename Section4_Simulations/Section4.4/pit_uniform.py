#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import math
import warnings
import numpy as np
import pandas as pd
from timeit import default_timer as time
from numpy.linalg import slogdet
from scipy.special import erf
from scipy.stats import uniform, gamma, norm


# In[ ]:
def standard_n_pdf(x):
    """
    Given x, return the probability density of x of standard Normal N(0, 1**2)  
    """
    return np.exp(-x*x/2)/2.5066282746310002


def likelihood_uniform_intercept_order2(tt, N=1000, p=5, g=20, inc=5, ss = None, X=None, Z=None, y=None):
    beta = tt[0:p] 
    #sigma = tt[-1]
    sigma = 1.5
    a = -beta[0]
    b = 2*beta[0]
    #myclip_a = -beta[0]
    #myclip_b = beta[0]
    #a, b = (myclip_a - 0) / eta, (myclip_b - 0) / eta
    ########### the first approximation
    tem = (X@beta[:, np.newaxis]).ravel() 
    #V = np.sqrt(Z.ravel()*Z.ravel()*beta[0]*beta[0]/3 + sigma*sigma)
    ###########
    gamma1 = uniform.ppf(0.8413448, loc = a, scale = b)
    E1 = tem + (gamma1*Z).ravel() 
    Z_values1 = (y-E1)/sigma
    values1 = standard_n_pdf(Z_values1)*0.5/sigma
    #values1 = norm.pdf(Z_values1, loc=0, scale=1)*0.5/sigma

    gamma2 = uniform.ppf(0.1586553, loc = a, scale = b) 
    E2 = tem + (gamma2*Z).ravel() 
    Z_values2 = (y-E2)/sigma 
    values2 = standard_n_pdf(Z_values2)*0.5/sigma
    #values2 = norm.pdf(Z_values2, loc=0, scale=1)*0.5/sigma
    #final = 0
    #den = values1 + values2
    #values1_final = np.where((values1 <= 0), )
    #den1 = np.where(den<=0, 1, den)
    #final = np.log(den1)
    final = 0
    for i in range(g):
        f1 = np.prod(values1[ss[i]:(ss[i]+inc)])
        f2 = np.prod(values2[ss[i]:(ss[i]+inc)])
        f = f1+f2 
        #f_t = np.where(np.isnan(f)|(f==np.inf)|(f==-np.inf)|(f<=0), 1, f)
        f_t = np.where(np.isnan(f)|(f<=0), 1, f)
        final = final + np.log(f_t)
    #final = (values1[np.newaxis, :]@m_vector)[0, 0] + (values2[np.newaxis, :]@m_vector)[0, 0]
    #final = np.sum(np.log(values1*0.5 + values1*0.5))
    return -1*final


def likelihood_uniform_intercept_order4(tt, N=1000, p=5, g=20, inc=5, ss = None, X=None, Z=None, y=None):
    beta = tt[0:p] 
    #sigma = tt[-1]
    sigma=1.5
    #sigma = tt[-1]
    #a = -beta[0]
    #b = 2*beta[0]
    #myclip_a = -beta[0]
    #myclip_b = beta[0]
    #a, b = (myclip_a - 0) / eta, (myclip_b - 0) / eta
    ########### the first approximation
    tem = (X@beta[:, np.newaxis]).ravel() 
    #V = np.sqrt(Z.ravel()*Z.ravel()*beta[0]*beta[0]/3 + sigma*sigma)
    ###########
    gamma1 = uniform.ppf(0.009787026, loc = -beta[0], scale = 2*beta[0])
    E1 = tem + gamma1*Z.ravel() 
    Z_values1 = (y-E1)/sigma
    values1 = standard_n_pdf(Z_values1)*0.04587583/sigma

    gamma2 = uniform.ppf(0.990213, loc = -beta[0], scale = 2*beta[0]) 
    E2 = tem + gamma2*Z.ravel() 
    Z_values2 = (y-E2)/sigma 
    values2 = standard_n_pdf(Z_values2)*0.04587583/sigma

    gamma3 = uniform.ppf(0.2290545, loc = -beta[0], scale = 2*beta[0]) 
    E3 = tem + gamma3*Z.ravel() 
    Z_values3 = (y-E3)/sigma 
    values3 = standard_n_pdf(Z_values3)*0.4541241/sigma

    gamma4 = uniform.ppf(0.7709455, loc = -beta[0], scale = 2*beta[0]) 
    E4 = tem + gamma4*Z.ravel() 
    Z_values4 = (y-E4)/sigma 
    values4 = standard_n_pdf(Z_values4)*0.4541241/sigma

    final = 0
    for i in range(g):
        f1 = np.prod(values1[ss[i]:(ss[i]+inc)])
        f2 = np.prod(values2[ss[i]:(ss[i]+inc)])
        f3 = np.prod(values3[ss[i]:(ss[i]+inc)])
        f4 = np.prod(values4[ss[i]:(ss[i]+inc)])
        f = f1+f2+f3+f4
        f_t = np.where(np.isnan(f)|(f==np.inf)|(f==-np.inf)|(f<=0), 1, f)  
        final = final + np.log(f_t)
    #final = 0
    #den = values1 + values2 + values3 + values4
    #den1 = np.where(den<=0, 1e-315, den)
    #final = np.log(den1)
    #for i in range(g):
    #    f1 = np.prod(values1[ss[i]:(ss[i]+inc)])
    #    f2 = np.prod(values2[ss[i]:(ss[i]+inc)])
    #    f = f1+f2 
    #    final = final + np.log(f)
    #final = (values1[np.newaxis, :]@m_vector)[0, 0] + (values2[np.newaxis, :]@m_vector)[0, 0]
    #final = np.sum(np.log(values1*0.5 + values1*0.5))
    #return -1*np.sum(final)
    return -1*final 

def likelihood_uniform_intercept_order10(tt, N=1000, p=5, g=20, inc=5, ss = None, X=None, Z=None, y=None):
    beta = tt[0:p] 
    #sigma = 1.5
    sigma = tt[-1]
    #sigma = tt[-1]
    #a = -beta[0]
    #b = 2*beta[0]
    #myclip_a = -beta[0]
    #myclip_b = beta[0]
    #a, b = (myclip_a - 0) / eta, (myclip_b - 0) / eta
    ########### the first approximation
    tem = (X@beta[:, np.newaxis]).ravel() 
    #V = np.sqrt(Z.ravel()*Z.ravel()*beta[0]*beta[0]/3 + sigma*sigma)
    ###########
    gamma1 = uniform.ppf(0.6861384, loc = -beta[0], scale = 2*beta[0])
    E1 = tem + gamma1*Z.ravel() 
    Z_values1 = (y-E1)/sigma
    values1 = standard_n_pdf(Z_values1)*0.3446425/sigma

    gamma2 = uniform.ppf(0.3138616, loc = -beta[0], scale = 2*beta[0]) 
    E2 = tem + gamma2*Z.ravel() 
    Z_values2 = (y-E2)/sigma 
    values2 = standard_n_pdf(Z_values2)*0.3446425/sigma

    gamma3 = uniform.ppf(0.9286742, loc = -beta[0], scale = 2*beta[0]) 
    E3 = tem + gamma3*Z.ravel() 
    Z_values3 = (y-E3)/sigma 
    values3 = standard_n_pdf(Z_values3)*0.1354837/sigma

    gamma4 = uniform.ppf(0.07132579, loc = -beta[0], scale = 2*beta[0]) 
    E4 = tem + gamma4*Z.ravel() 
    Z_values4 = (y-E4)/sigma 
    values4 = standard_n_pdf(Z_values4)*0.1354837/sigma

    gamma5 = uniform.ppf(0.9935101, loc = -beta[0], scale = 2*beta[0]) 
    E5 = tem + gamma5*Z.ravel() 
    Z_values5 = (y-E5)/sigma 
    values5 = standard_n_pdf(Z_values5)*0.01911158/sigma

    gamma6 = uniform.ppf(0.006489943, loc = -beta[0], scale = 2*beta[0]) 
    E6 = tem + gamma6*Z.ravel() 
    Z_values6 = (y-E6)/sigma 
    values6 = standard_n_pdf(Z_values6)*0.01911158/sigma

    gamma7 = uniform.ppf(0.9998294, loc = -beta[0], scale = 2*beta[0]) 
    E7 = tem + gamma7*Z.ravel() 
    Z_values7 = (y-E7)/sigma 
    values7 = standard_n_pdf(Z_values7)*0.0007580711/sigma

    gamma8 = uniform.ppf(0.0001706037, loc = -beta[0], scale = 2*beta[0]) 
    E8 = tem + gamma8*Z.ravel() 
    Z_values8 = (y-E8)/sigma 
    values8 = standard_n_pdf(Z_values8)*0.0007580711/sigma

    gamma9 = uniform.ppf(0.9999994, loc = -beta[0], scale = 2*beta[0]) 
    E9 = tem + gamma9*Z.ravel() 
    Z_values9 = (y-E9)/sigma 
    values9 = standard_n_pdf(Z_values9)*4.310651e-06/sigma

    gamma10 = uniform.ppf(5.885197e-07, loc = -beta[0], scale = 2*beta[0]) 
    E10 = tem + gamma10*Z.ravel() 
    Z_values10 = (y-E10)/sigma 
    values10 = standard_n_pdf(Z_values10)*4.310651e-06/sigma


    final = 0
    for i in range(g):
        f1 = np.prod(values1[ss[i]:(ss[i]+inc)])
        f2 = np.prod(values2[ss[i]:(ss[i]+inc)])
        f3 = np.prod(values3[ss[i]:(ss[i]+inc)])
        f4 = np.prod(values4[ss[i]:(ss[i]+inc)])
        f5 = np.prod(values5[ss[i]:(ss[i]+inc)])
        f6 = np.prod(values6[ss[i]:(ss[i]+inc)])
        f7 = np.prod(values7[ss[i]:(ss[i]+inc)])
        f8 = np.prod(values8[ss[i]:(ss[i]+inc)])
        f9 = np.prod(values9[ss[i]:(ss[i]+inc)])
        f10 = np.prod(values10[ss[i]:(ss[i]+inc)])
        f = f1+f2+f3+f4+f5+f6+f7+f8+f9+f10  
        f_t = np.where(np.isnan(f)|(f==np.inf)|(f==-np.inf)|(f<=0), 1, f)  
        final = final + np.log(f_t)
    #final = 0
    #den = values1 + values2 + values3 + values4
    #den1 = np.where(den<=0, 1e-315, den)
    #final = np.log(den1)
    #for i in range(g):
    #    f1 = np.prod(values1[ss[i]:(ss[i]+inc)])
    #    f2 = np.prod(values2[ss[i]:(ss[i]+inc)])
    #    f = f1+f2 
    #    final = final + np.log(f)
    #final = (values1[np.newaxis, :]@m_vector)[0, 0] + (values2[np.newaxis, :]@m_vector)[0, 0]
    #final = np.sum(np.log(values1*0.5 + values1*0.5))
    #return -1*np.sum(final)
    return -1*final



def likelihood_uniform_intercept_order4_withsigma(tt, N=1000, p=5, g=20, inc=5, ss = None, X=None, Z=None, y=None):
    beta = tt[0:p] 
    sigma = tt[-1]
    #sigma = tt[-1]
    #a = -beta[0]
    #b = 2*beta[0]
    #myclip_a = -beta[0]
    #myclip_b = beta[0]
    #a, b = (myclip_a - 0) / eta, (myclip_b - 0) / eta
    ########### the first approximation
    tem = (X@beta[:, np.newaxis]).ravel() 
    #V = np.sqrt(Z.ravel()*Z.ravel()*beta[0]*beta[0]/3 + sigma*sigma)
    ###########
    gamma1 = uniform.ppf(0.009787026, loc = -beta[0], scale = 2*beta[0])
    E1 = tem + gamma1*Z.ravel() 
    Z_values1 = (y-E1)/sigma
    values1 = standard_n_pdf(Z_values1)*0.04587583

    gamma2 = uniform.ppf(0.990213, loc = -beta[0], scale = 2*beta[0]) 
    E2 = tem + gamma2*Z.ravel() 
    Z_values2 = (y-E2)/sigma 
    values2 = standard_n_pdf(Z_values2)*0.04587583

    gamma3 = uniform.ppf(0.2290545, loc = -beta[0], scale = 2*beta[0]) 
    E3 = tem + gamma3*Z.ravel() 
    Z_values3 = (y-E3)/sigma 
    values3 = standard_n_pdf(Z_values3)*0.4541241

    gamma4 = uniform.ppf(0.7709455, loc = -beta[0], scale = 2*beta[0]) 
    E4 = tem + gamma4*Z.ravel() 
    Z_values4 = (y-E4)/sigma 
    values4 = standard_n_pdf(Z_values4)*0.4541241

    #final = 0
    #for i in range(g):
    #    f1 = np.prod(values1[ss[i]:(ss[i]+inc)])
    #    f2 = np.prod(values2[ss[i]:(ss[i]+inc)])
    #    f3 = np.prod(values3[ss[i]:(ss[i]+inc)])
    #    f4 = np.prod(values4[ss[i]:(ss[i]+inc)])
    #    f = f1+f2+f3+f4  
    #    final = final + np.log(f)
    #final = 0
    den = values1 + values2 + values3 + values4
    den1 = np.where(den<=0, 1e-315, den)
    final = np.log(den1)
    #for i in range(g):
    #    f1 = np.prod(values1[ss[i]:(ss[i]+inc)])
    #    f2 = np.prod(values2[ss[i]:(ss[i]+inc)])
    #    f = f1+f2 
    #    final = final + np.log(f)
    #final = (values1[np.newaxis, :]@m_vector)[0, 0] + (values2[np.newaxis, :]@m_vector)[0, 0]
    #final = np.sum(np.log(values1*0.5 + values1*0.5))
    return -1*np.sum(final)


def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)



