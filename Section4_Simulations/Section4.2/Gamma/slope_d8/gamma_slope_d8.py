#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
import time
import os 
from scipy.optimize import LinearConstraint
from scipy.optimize import BFGS
from scipy.optimize import minimize, fsolve
from scipy.optimize import NonlinearConstraint
from scipy.linalg import block_diag
from scipy.stats import uniform, gamma
from gamma_sadd3 import gamma_sda, equlity_constraints_gamma, RF_step2_gamma, get_near_psd  


# In[2]:
print(os.getcwd())
M = 10 #### number of repeats for the number. 
repeats = 5  #### # of repeats for initial values. 
k = 2 ##### dimension of Z
g = 20 ##### # of groups
tol = 1e-05
# In[2]:
d_matrix = np.array([8])
N_matrix = np.array([100, 200, 500])
# In[5]:
for d in d_matrix:
    for N in N_matrix:
        n = d
        if d == 8:
            #true = np.array([1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 0.3])
            true = np.array([1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 0.1, 1.0, 0.8])
            RF_index = np.array(['Y', 'Y', 'N', 'N', 'N', 'N', 'N', 'N'])
        else:
            #true = np.array([1.0, 1.5, 1.0, 1.5, 1.0, 0.3])
            true = np.array([1.0, 1.0, 1.5, 1.0, 1.5, 0.1, 1.0, 0.8])
            #true = np.array([1.0, 1.0, 1.5, 1.0, 1.5, 0.1, 1.0, 0.5])
            RF_index = np.array(['Y', 'Y', 'N', 'N', 'N'])

        
        if N == 100:
            maxiter_my = 2000
        elif N == 200:
            maxiter_my = 3000
        else:
            maxiter_my = 3000    
            
        containers = np.zeros((M, len(true)+k*g+3))
        RF_true = np.where(RF_index == 'Y')[0].astype(int)
        m_vector = np.ones(len(RF_true))[:, np.newaxis]
        #containers = np.zeros((M, len(true)+g+4)) ##### grand matrix to contain the results
        ss = int(N/g) ##### group size
        group_index = np.repeat(np.linspace(1, g, g), ss).astype(int)
        unique_levels = len(np.unique(group_index))

        #### get the dummy matrix
        pd_group_index = pd.DataFrame(group_index, columns = ['group_index'])
        dummy_matrix = pd.get_dummies(pd_group_index['group_index'].astype('category',copy=False), prefix='x0')

        ###### get the true parameters
        beta_v = true[0:n] ### first n are beta 
        sigma = true[n:(n+1)]
        b_v = true[(n+1):(n+1+k)]
        #b_v = true[(n+1+k):(n+1+2*k)];
        #sigma = true[(n+2*k):(n+2*k+1)];
        #shape_my = 10
        #rate_my = 10
        #lambda_v = true[n: (n+k)] #### the following k are lambda 
        #sigma = true[-1] #### the following 1 is sigma 

        ##### generate true gamma
        #np.random.seed(2019)
        #gamma = np.random.gamma(shape=a_v, scale=1/b_v, size = g)
        #gamma = np.random.uniform(low = -abs(beta_v[0]), high = abs(beta_v[0]), size = g)
        np.random.seed(2019)
        gamma1 = np.random.gamma(shape=1, scale=1/b_v[0], size = g)
        np.random.seed(2019)
        gamma2 = np.random.gamma(shape=1, scale=1/b_v[1], size = g)
        #####
        gamma_h = np.concatenate((gamma1[:, np.newaxis], gamma2[:, np.newaxis]), 1)
        gamma_v = np.concatenate((gamma1[:, np.newaxis], gamma2[:, np.newaxis]), 0)
        #gamma = np.random.gamma(shape=shape_my, scale=1/rate_my, size = g)
        #gamma = np.random.uniform(low = -abs(beta_v[0]), high = abs(beta_v[0]), size = g)
        if d==5:
            mean = np.array([1.0, 1.5, 1.0, 1.5])
        else:
            mean = np.array([1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0])

        for t in range(M):
            np.random.seed(2020+7*t)
            cov = np.random.uniform(0, 0.2, len(mean)**2).reshape(len(mean), len(mean))
            np.fill_diagonal(cov, 1) 
            cov1 = get_near_psd(cov)

            np.random.seed(2020+7*t)
            X1 = np.abs(np.random.multivariate_normal(mean, cov1, size=N))
            X = np.concatenate((np.ones(N)[:, np.newaxis], X1), 1)
            #np.corrcoef(X1.T)
            Z = X[:, 0:2]
            E = (X@beta_v[:, np.newaxis]).ravel() + (Z*np.repeat(gamma_h, int(N/g), axis=0)).sum(axis =1)
            np.random.seed(2020+7*t)
            Y = np.random.normal(loc=E, scale=np.repeat(sigma, N))
            #Z = X[:, 0][:, np.newaxis]
            #E = (X@beta_v[:, np.newaxis]).ravel() + Z.ravel()*np.repeat(gamma, int(N/g))
            #np.random.seed(2020+7*t)
            #Y = np.random.normal(loc=E, scale=np.repeat(sigma, N))

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
            
            #### the main matrix left right 
            ######### first linear constraints 
            #tem1 = np.diag(np.concatenate((np.ones(len(true)), -1*Z[:,0])))
            #tem1 = tem1[(d+1+k+k):(d+1+k+k+N), d+1+k] = np.ones(N)
            tem1 = np.diag(np.concatenate((np.ones(len(true)), -1*Z[:,0])))
            tem1[(d+1+k):(d+1+k+N), d+1] = np.ones(N)
            ######## second linear constraints 
            tem2 = np.diag(np.concatenate((np.ones(len(true)), -1*Z[:,1])))
            tem2[(d+1+k):(d+1+k+N), (d+k)] = np.ones(N)
            #########
            lb = [tol]*d+ [-100]*1 + [tol]*k + [-100]*N 
            ub = [100]*(d+1+k+N)
            sign_constraint1 = LinearConstraint(tem1, lb, ub, keep_feasible=True)
            #lb_gg = [-100]*(d+1+k+k) + [tol]*N 
            #ub_gg = [100]*(d+1+k+k+N) 
            sign_constraint2 = LinearConstraint(tem2, lb, ub, keep_feasible=True)


            ######### 
            cons = equlity_constraints_gamma(tol, RF_index, X, Y, Z)
            lb_e = Y; ub_e = Y
            equality_constraint = NonlinearConstraint(cons.gradient_K_fun_gamma, lb_e, ub_e)

            ########
            #shasha = '/data/Nchen/saddle/gamma1/'
            #name = '_'.join(['Gamma_slope', 'd', str(d), 'N', str(N)])
            #final_name = ''.join([name, '.csv'])
            #final_name1 = ''.join((shasha, final_name))  

            #iniseed = np.array([2017, 1989, 1990, 1991, 2077])
            #iniseed = np.array([2016, 1989, 1979, 2088, 2029])
            iniseed = np.array([544, 1063, 1133, 1807, 2552])
            #taicha = np.zeros(3000)
            #iniseed =  np.linspace(1, 3000, num=3000).astype(int)
            initials = np.zeros((len(iniseed), d+k+1+N))
            for j in range(len(iniseed)):
                if d==5:
                    tem0 = uniform.rvs(loc=0.9, scale=1.2, size=1, random_state=iniseed[j])
                    tem1 = uniform.rvs(loc=0.9, scale=1.2, size=1, random_state=iniseed[j])
                    tem2 = uniform.rvs(loc=1.2, scale=1.7, size=1, random_state=iniseed[j])
                    tem3 = uniform.rvs(loc=0.8, scale=1.2, size=1, random_state=iniseed[j])
                    tem4 = uniform.rvs(loc=1.2, scale=1.7, size=1, random_state=iniseed[j])
                    baba = np.array([tem0, tem1, tem2, tem3, tem4]).ravel()
                else:
                    tem0 = uniform.rvs(loc=0.9, scale=1.2, size=1, random_state=iniseed[j])
                    tem1 = uniform.rvs(loc=0.9, scale=1.2, size=1, random_state=iniseed[j])
                    tem2 = uniform.rvs(loc=1.2, scale=1.7, size=1, random_state=iniseed[j])
                    tem3 = uniform.rvs(loc=0.8, scale=1.2, size=1, random_state=iniseed[j])
                    tem4 = uniform.rvs(loc=1.2, scale=1.7, size=1, random_state=iniseed[j])
                    tem5 = uniform.rvs(loc=0.8, scale=1.2, size=1, random_state=iniseed[j])
                    tem6 = uniform.rvs(loc=1.2, scale=1.7, size=1, random_state=iniseed[j])
                    tem7 = uniform.rvs(loc=0.8, scale=1.2, size=1, random_state=iniseed[j])
                    baba = np.array([tem0, tem1, tem2, tem3, tem4, tem5, tem6, tem7]).ravel()

                tem8 = uniform.rvs(loc=0.0, scale=0.3, size=1, random_state=iniseed[j]) 
                tem9 = uniform.rvs(loc=0.8, scale=1.2, size=1, random_state=iniseed[j])
                tem10 = uniform.rvs(loc=0.7, scale=1.0, size=1, random_state=iniseed[j])
                baba1 = np.array([tem9, tem10]).ravel()
                #tem4 = uniform.rvs(loc=0.5, scale=1.5, size=k, random_state=iniseed[j])
                #tem3 = true[(d+k+k):(d+k+k+1)] + uniform.rvs(loc=-0.05, scale=0.1, size=1, random_state=iniseed[j]) 
                t_start = uniform.rvs(loc=-0.18, scale=0.36, size=N, random_state=iniseed[j])
                initials[j, :] = np.concatenate((baba, np.array(tem8), baba1, t_start))
                #taicha[j] = np.sqrt(np.sum((initials[j, 0:d+1+k+k] - true)**2)/len(true))
            ####### get uniform object 
            uni_obj = gamma_sda(tol, g, RF_index, X, Y, Z)

            

            ########
            chacha = np.inf 
            x_final = None
            x_final_candidate = None
            success = False
            ite_grand = 0
            for j in range(len(iniseed)):
                x0_t = initials[j, :].ravel()
                meth1 = minimize(fun = uni_obj.LK_negative, x0 = x0_t, method='trust-constr', constraints = [sign_constraint1, sign_constraint2, equality_constraint], options={'maxiter': maxiter_my})
                #if meth1.success:
                l1 = len(np.where((meth1.x[0:d] > 0) == False)[0])
                l2 = len(np.where((meth1.x[(d+1):(d+1+k)] > 0) == False)[0])
                if ((l1<1) & (l2<1)):
                    x_final_candidate = np.concatenate((meth1.x[0:d], np.abs([meth1.x[d]]), meth1.x[(d+1):]))
                    if ((meth1.fun <= chacha) & (meth1.fun > -1000)):
                        chacha = meth1.fun 
                        success = True
                        ite_grand = meth1.niter
                        x_final = x_final_candidate
                        print('updated')

                if (j == len(iniseed)-1) & (success == False):
                    x_final = x_final_candidate
                    print('The last run, use the most possible result')
                print(j)
                print(meth1.success)
                print(meth1.niter)
                print(meth1.fun)
                print(chacha)
                print(meth1.x[0:(d+k+1)])
            # In[14]:
            tem1 = {'results': x_final, 'iteration': ite_grand, 'LK_negative': chacha, 'Success': success}
            res = x_final
            print(tem1)
            
            #print(tem1)
            #res = tem1['results']
            if res is None:
                fxs = np.zeros(d+k+1)
                rfs = np.zeros(g*k)
            else:
                fxs = res[0:(d+k+1)] 
                ######## do the step 2 optimization 
                ######## do the step 2 optimization 
                x0_rf = uniform.rvs(loc=0.1, scale=3.0, size=k*g, random_state=2021)
                equ_rf = np.diag(np.full(k*g,1))  
                ########## Create equality constraints
                lb_rf = [tol]*g*k
                ub_rf = [100]*g*k
                #ub_rf = [np.inf, np.inf, -1e-05, -1e-05]
                linear_constraint_rf = LinearConstraint(equ_rf, lb_rf, ub_rf, keep_feasible=True)
                meth2 = minimize(fun = RF_step2_gamma, x0 = x0_rf, args = (res[0:d], res[d:(d+1)], np.array([1, 1]), res[(d+1):(d+1+k)], k, g, m_vector, X, Y, Z_long), method='trust-constr', constraints = [linear_constraint_rf], options = {'xtol': 1e-10, 'gtol':1e-10, 'maxiter': maxiter_my})

                #meth2 = minimize(fun = RF_step2_gamma, x0 = x0_rf, args = (res[0:d], res[d:(d+k)], res[(d+k):(d+k+k)], res[(d+k+k):(d+k+k+1)], k, g, m_vector, X, Y, Z_long), method='trust-constr', constraints = [linear_constraint_rf], options = {'xtol': 1e-10, 'gtol':1e-10, 'maxiter': 2000})
                rfs = meth2.x
            hh = np.concatenate((fxs, [tem1['Success']], [tem1['LK_negative']], [tem1['iteration']], rfs)) 
            containers[t, :] = hh
            print(t)

        print('Run finished. Write out results.')
        tjj = pd.DataFrame(containers)
        shasha = '/data/Nchen/saddle/gamma_d8/'
        name = '_'.join(['Gamma_slope', 'd', str(d), 'N', str(N)])
        final_name = ''.join([name, '.csv'])
        final_name1 = ''.join((shasha, final_name))     
        # In[8]:
        tjj.to_csv(final_name1)