#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
import time
from scipy.optimize import LinearConstraint
from scipy.optimize import BFGS
from scipy.optimize import minimize, fsolve
from scipy.optimize import NonlinearConstraint
from scipy.linalg import block_diag
from scipy.stats import uniform
from pit_uniform import likelihood_uniform_intercept_order2, likelihood_uniform_intercept_order4, get_near_psd 


# In[2]:


#np.random.seed(2019)
#gamma = np.random.laplace(0, 0.1, size = 20)
#gamma


# In[3]:


M = 10 #### number of repeats for the number. 
repeats = 5  #### # of repeats for initial values. 
k = 1 ##### dimension of Z
g = 20 ##### # of groups
tol = 1e-05
# In[2]:
#d_matrix = np.array([5])
#N_matrix = np.array([100, 200])


# In[4]:


d_matrix = np.array([8])
N_matrix = np.array([100, 200, 500])
final_results = np.zeros(3)
rr = 0


# In[5]:


for d in d_matrix:
    for N in N_matrix:
        n = d
        if d == 8:
            true = np.array([1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5])
            RF_index = np.array(['Y', 'N', 'N', 'N', 'N', 'N', 'N', 'N'])
        else:
            true = np.array([1.0, 1.0, 1.5, 1.0, 1.5, 1.5])
            RF_index = np.array(['Y', 'N', 'N', 'N', 'N'])

        #containers = np.zeros((M, len(true)+k*g+3))
        #containers = np.zeros((M, len(true)+g+4)) ##### grand matrix to contain the results
        ss = int(N/g) ##### group size
        group_index = np.repeat(np.linspace(1, g, g), ss).astype(int)
        unique_levels = len(np.unique(group_index))

        #### get the dummy matrix
        pd_group_index = pd.DataFrame(group_index, columns = ['group_index'])
        dummy_matrix = pd.get_dummies(pd_group_index['group_index'].astype('category',copy=False), prefix='x0')

        ###### get the true parameters
        beta_v = true[0:n] ### first n are beta 
        #print(beta_v)
        #lambda_v = true[n: (n+k)] #### the following k are lambda 
        sigma = true[-1] #### the following 1 is sigma 
        #print(sigma)

        ##### generate true gamma
        np.random.seed(2019)
        gamma = np.random.uniform(low = -abs(beta_v[0]), high = abs(beta_v[0]), size = g)

        if d==5:
            mean = np.array([1.0, 1.5, 1.0, 1.5])
        else:
            mean = np.array([1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0])

        ss = np.linspace(start = 0, stop = N, num=g, endpoint=False).astype(int)

        rmse_m = np.zeros(M)
        for t in range(M):
            np.random.seed(2020+7*t)
            cov = np.random.uniform(0, 0.2, len(mean)**2).reshape(len(mean), len(mean))
            np.fill_diagonal(cov, 1) 
            cov1 = get_near_psd(cov)

            np.random.seed(2020+7*t)
            X1 = np.abs(np.random.multivariate_normal(mean, cov1, size=N))
            X = np.concatenate((np.ones(N)[:, np.newaxis], X1), 1)
            #np.corrcoef(X1.T)
            Z = X[:, 0][:, np.newaxis]
            E = (X@beta_v[:, np.newaxis]).ravel() + Z.ravel()*np.repeat(gamma, int(N/g))
            np.random.seed(2020+7*t)
            Y = np.random.normal(loc=E, scale=np.repeat(sigma, N))

            ####################### get Z_long
            RF_true = np.where(RF_index == 'Y')[0].astype(int)
            X_df = pd.DataFrame(X)
            X_df_subset = X_df.iloc[:, RF_true]

            ##### Create the final Z matrix using a for loop
            Z1 = pd.DataFrame()
            for j in range(X_df_subset.shape[1]):
                tem = pd.DataFrame(X_df_subset.iloc[:, j].values[:, np.newaxis]*dummy_matrix.values)
                Z1 = pd.concat([Z1, tem], axis=1)
            Z_long = Z1.values
            ###Data is ready X, Z, Y

            ###### uniform bound constraints 
            equ = np.zeros((len(true)-1, len(true)-1))
            np.fill_diagonal(equ, 1)
            lb = [tol]*(len(true)-1)
            ub = [100]*(len(true)-1)
            sign_constraint = LinearConstraint(equ, lb, ub, keep_feasible=True)

            ######
            iniseed = np.array([2017, 1989, 1990, 1991, 2077])
            initials = np.zeros((len(iniseed), d))
            for j in range(len(iniseed)):
                tem1 = uniform.rvs(loc=1.0, scale=1.0, size=d, random_state=int(iniseed[j])) 
                #tem2 = uniform.rvs(loc=0.0, scale=1.0, size=1, random_state=int(iniseed[j]))
                #tem2 = uniform.rvs(loc=0.0, scale=1.0, size=1, random_state=int(iniseed[j])) 
                #tem1 = true[0:d] + uniform.rvs(loc=-0.1, scale=0.2, size=d, random_state=iniseed[j])
                #tem2 = true[d:(d+1)] + uniform.rvs(loc=-0.05, scale=0.10, size=1, random_state=iniseed[j])
                #t_start = uniform.rvs(loc=-1, scale=2, size=N, random_state=iniseed[j])
                #initials[j, :] = np.concatenate((tem1))
                #initials[j, :] = np.concatenate((tem1, np.array(tem2)))
                initials[j, :] = tem1

            ###########
            chacha = np.inf 
            x_final = None 
            success = False
            ite_grand = 0
            for j in range(len(iniseed)):
                x0_t = initials[j, :].ravel()
                meth1 = minimize(fun = likelihood_uniform_intercept_order2, 
                                 args = (N, d, g, int(N/g), ss, X, Z, Y), 
                                 x0 = x0_t, method='trust-constr', 
                                 constraints = [sign_constraint], options={'maxiter': 2000})
                #if meth1.success:
                    #success = True
                if ((meth1.fun <= chacha) & (meth1.fun > -1000000)):
                    x_final = meth1.x
                    #x_final = np.concatenate((meth1.x[0:d], np.array(np.abs([meth1.x[d]])), meth1.x[(d+1):]))
                    chacha = meth1.fun
                    ite_grand = meth1.niter
                    #print('updated')
                #print(j)
                #print(meth1.success)
                #print(meth1.niter)
                #print(meth1.fun)
                #print(chacha)
                #print(meth1.x)
            #print(t)
            #print(d)
            #print(x_final)
            #print(beta_v)
            rmse1 = np.sqrt(np.sum((x_final[0:d] - beta_v[0:d])**2)/len(beta_v[0:d]))
            #print(rmse1)
            rmse_m[t] = rmse1
        final_results[rr] = np.median(rmse_m)
        #print(rr)
        print(d)
        print(N)
        print(np.median(rmse_m))
        rr = rr + 1


# In[6]:


#np.median(rmse_m)


# In[7]:


#beta_v


# In[8]:


#meth1.x


# In[9]:


#mama = pd.DataFrame(final_results[np.newaxis, :])


# In[10]:


mama = pd.DataFrame(final_results[np.newaxis, :], columns = ['d=8, n=100', 'd=8, n=200', 'd=8, n=500'])


# In[11]:


#mama = pd.DataFrame(final_results[np.newaxis, :], columns = ['d=5, n=100', 'd=5, n=200', 'd=5, n=500',
#                                             'd=8, n=100', 'd=8, n=200', 'd=8, n=500'])


# In[12]:


mama.to_csv('uniform_PIT_Q2_d8.csv')

