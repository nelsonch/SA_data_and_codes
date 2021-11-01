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
from scipy.stats import uniform, triang
from triangular_sadd import triangular_sda, equlity_constraints_triangular, step2_RF, get_near_psd, get_dummy_own


# In[2]:


M = 10 #### number of repeats for the number. 
repeats = 5  #### # of repeats for initial values. 
k = 2 ##### dimension of Z
g = 20 ##### # of groups
tol = 1e-05
# In[2]:
d_matrix = np.array([5])
N_matrix = np.array([100, 200, 500])


# In[3]:


#d = d_matrix[0]
#N = N_matrix[0]
#t = 0
# In[4]:

for d in d_matrix:
    for N in N_matrix:
        n = d
        if d == 8:
            true = np.array([2.5, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.1])
            RF_index = np.array(['Y', 'Y', 'N', 'N', 'N', 'N', 'N', 'N'])
        else:
            true = np.array([2.5, 1.0, 1.5, 1.5, 1.5, 0.1])
            RF_index = np.array(['Y', 'Y', 'N', 'N', 'N'])


        containers = np.zeros((M, len(true)+k*g+3))
        #containers = np.zeros((M, len(true)+g+4)) ##### grand matrix to contain the results
        ss = int(N/g) ##### group size
        group_index = np.repeat(np.linspace(1, g, g), ss).astype(int)
        unique_levels = len(np.unique(group_index))

        #### get the dummy matrix
        pd_group_index = pd.DataFrame(group_index, columns = ['group_index'])
        dummy_matrix = pd.get_dummies(pd_group_index['group_index'].astype('category',copy=False), prefix='x0')

        ###### get the true parameters
        beta_v = true[0:n] ### first n are beta 
        #lambda_v = true[n: (n+k)] #### the following k are lambda 
        sigma = true[-1] #### the following 1 is sigma 

        # In[5]:
        #seeds = np.linspace(0, int(k*g), num=int(k*g), endpoint=False)
        #gamma = np.zeros(len(seeds))
        #for i in range(len(seeds)):
        #    gamma[i] = triang.rvs(c=0.5, loc=-beta_v[0], scale=2*beta_v[0], size=1, random_state=int(seeds[i])) 
        #gamma1 = triang.rvs(c=0.5, loc=-beta_v[0], scale=2*beta_v[0], size=g, random_state=2020)
        #gamma2 = triang.rvs(c=0.5, loc=-beta_v[1], scale=2*beta_v[1], size=g, random_state=2020)

        seeds = np.linspace(0, 20, num=20, endpoint=False)
        gamma1 = np.zeros(len(seeds))
        gamma2 = gamma1
        for i in range(len(seeds)):
            gamma1[i] = triang.rvs(c=0.5, loc=-beta_v[0], scale=2*beta_v[0], size=1, random_state=int(seeds[i]))
            gamma2[i] = triang.rvs(c=0.5, loc=-beta_v[1], scale=2*beta_v[1], size=1, random_state=int(seeds[i]))
                
        gamma_h = np.concatenate((gamma1[:, np.newaxis], gamma2[:, np.newaxis]), 1)
        gamma_v = np.concatenate((gamma1[:, np.newaxis], gamma2[:, np.newaxis]), 0)
        #gamma_h = gamma.reshape(g, k, order='F')
        #gamma_v = gamma[:, np.newaxis]

        if d==5:
            mean = np.array([1.0, 1.0, 1.0, 1.0])
        else:
            mean = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        for t in range(M):
            np.random.seed(2020+7*t)
            cov = np.random.uniform(0, 0.2, len(mean)**2).reshape(len(mean), len(mean))
            np.fill_diagonal(cov, 1) 
            cov1 = get_near_psd(cov)

            np.random.seed(2020+7*t)
            X1 = np.abs(np.random.multivariate_normal(mean, cov1, size=N))
            X = np.concatenate((np.ones(N)[:, np.newaxis], X1), 1)
            #np.corrcoef(X1.T)
            Z = X[:, 0:k]
            #E = (X@beta_v[:, np.newaxis]).ravel() + Z.ravel()*np.repeat(gamma, int(N/g))
            bhj = np.repeat(gamma_h, int(N/g), axis=0)
            #print(bhj)
            E = (X@beta_v[:, np.newaxis]).ravel() + (Z*bhj).sum(axis =1)
            np.random.seed(2020+7*t)
            Y = np.random.normal(loc=E, scale=np.repeat(np.sqrt(sigma), N))

            ####################### get Z_long
            RF_true = np.where(RF_index == 'Y')[0].astype(int)
            X_df = pd.DataFrame(X)
            X_df_subset = X_df.iloc[:, RF_true]

            ##### Create the final Z matrix using a for loop
            baibai = np.zeros(N)[:, np.newaxis]
            for j in range(k):
                gght = get_dummy_own(Z[:, j], g=g)
                baibai = np.concatenate((baibai, gght), axis=1)
            Z_long = baibai[:, 1:]
            ###Data is ready X, Z, Y 

            ###### uniform bound constraints 
            equ = np.zeros((d+1+N, d+1+N))
            np.fill_diagonal(equ, 1)
            lb = [tol]*(d+1) + [-100]*N 
            ub = [100]*d + [100]*1 + [100]*N
            sign_constraint = LinearConstraint(equ, lb, ub, keep_feasible=True)


            ###### uniform K' = Y constraints
            eq_cons = equlity_constraints_triangular(tol, RF_index, X, Y, Z)
            lb_e = Y; ub_e = Y
            equality_constraint = NonlinearConstraint(eq_cons.gradient_K_fun_triangular, lb_e, ub_e)


            iniseed = np.array([2017, 1989, 1990, 1991, 2077])
            initials = np.zeros((len(iniseed), d+1+N))
            for j in range(len(iniseed)):
                if d==5:
                    tem0 = uniform.rvs(loc=beta_v[0], scale=0.4, size=1, random_state=int(iniseed[j]))
                    tem1 = uniform.rvs(loc=beta_v[1], scale=0.4, size=1, random_state=int(iniseed[j]))
                    tem2 = uniform.rvs(loc=beta_v[2], scale=0.4, size=1, random_state=int(iniseed[j]))
                    tem3 = uniform.rvs(loc=beta_v[3], scale=0.4, size=1, random_state=int(iniseed[j]))
                    #tem4 = uniform.rvs(loc=1.0, scale=0.4, size=1, random_state=int(iniseed[j]))
                    baba = np.array([tem0, tem1, tem2, tem3, tem3]).ravel()
                else:
                    tem0 = uniform.rvs(loc=beta_v[0], scale=0.4, size=1, random_state=int(iniseed[j]))
                    tem1 = uniform.rvs(loc=beta_v[1], scale=0.4, size=1, random_state=int(iniseed[j]))
                    tem2 = uniform.rvs(loc=beta_v[2], scale=0.4, size=1, random_state=int(iniseed[j]))
                    tem3 = uniform.rvs(loc=beta_v[3], scale=0.4, size=1, random_state=int(iniseed[j]))
                    #tem0 = uniform.rvs(loc=beta_v[0], scale=0.2, size=1, random_state=int(iniseed[j]))
                    #tem1 = uniform.rvs(loc=beta_v[1], scale=0.2, size=1, random_state=int(iniseed[j]))
                    #tem2 = uniform.rvs(loc=1.0, scale=0.4, size=1, random_state=int(iniseed[j]))
                    #tem3 = uniform.rvs(loc=1.0, scale=0.4, size=1, random_state=int(iniseed[j]))
                    #tem4 = uniform.rvs(loc=1.0, scale=0.4, size=1, random_state=int(iniseed[j]))
                    #tem5 = uniform.rvs(loc=1.0, scale=0.4, size=1, random_state=int(iniseed[j]))
                    #tem6 = uniform.rvs(loc=1.0, scale=0.4, size=1, random_state=int(iniseed[j]))
                    #tem7 = uniform.rvs(loc=1.0, scale=0.4, size=1, random_state=int(iniseed[j]))
                    #tem8 = uniform.rvs(loc=1.0, scale=0.4, size=1, random_state=int(iniseed[j]))
                    baba = np.array([tem0, tem1, tem2, tem3, tem3, tem3, tem3, tem3]).ravel()

                #tem1 = uniform.rvs(loc=1.0, scale=2.0, size=d-1, random_state=int(iniseed[j]))
                tem22 = uniform.rvs(loc=0.01, scale=0.05, size=1, random_state=int(iniseed[j])) 
                #tem1 = true[0:d] + uniform.rvs(loc=-0.1, scale=0.2, size=d, random_state=iniseed[j])
                #tem2 = true[d:(d+1)] + uniform.rvs(loc=-0.05, scale=0.10, size=1, random_state=iniseed[j])
                t_start = uniform.rvs(loc=-2.0, scale=2.0, size=N, random_state=iniseed[j])
                initials[j, :] = np.concatenate((baba, np.array(tem22), t_start))
            ####### get uniform object 
            uni_obj = triangular_sda(tol, g, RF_index, X, Y, Z) 
            #uni_obj.LK_negative(initials[0, :])


            chacha = np.inf 
            x_final = None 
            success = False
            ite_grand = 0
            for j in range(len(iniseed)):
                x0_t = initials[j, :].ravel()
                meth1 = minimize(fun = uni_obj.LK_negative, x0 = x0_t, method='trust-constr', constraints = [sign_constraint, equality_constraint], options={'maxiter': 2000})
                if meth1.success:
                    success = True
                    if ((meth1.fun <= chacha) & (meth1.fun > -1000)):
                        if len(np.where((meth1.x[0:d] > 0) == False)[0]) < 1:
                            x_final = np.concatenate((meth1.x[0:d], np.array(np.abs([meth1.x[d]])), meth1.x[(d+1):]))
                            chacha = meth1.fun
                            ite_grand = meth1.niter
                            print('updated')
                print(j)
                print(meth1.success)
                print(meth1.niter)
                print(meth1.fun)
                print(chacha)
                print(meth1.x[0:(d+1)])



            tem1 = {'results': x_final, 'iteration': ite_grand, 'LK_negative': chacha, 'Success': success}
            print(tem1)
            res = x_final

            fxs = res[0:(d+1)] 
            # In[18]:
            # In[15]:
            #print(tem1)
            # In[15]:
            #print(tem1)
            res = tem1['results']
            if res is None:
                fxs = np.zeros(d+1)
                rfs = np.zeros(g*k)
            else:
                fxs = res[0:(d+1)] 
                ######## do the step 2 optimization 
                sigma_hat = fxs[-1]
                #if fxs[-1] <= 1e-03:
                #    sigma_hat = 0.1
                #    print('estimate')
                #print(sigma_hat)
                if sigma_hat <= 0.001:
                    sigma_hat = 1
                    print('sigma_hat corrected')
                ###### 2 linear constraints 
                #x0_rf = triang.rvs(c=0.5, loc=-res[0], scale=2*res[0]-0.011, size=g, random_state=2020)
                x0_rf = np.zeros(g)
                v0 = np.abs(x0_rf)+0.01
                #v1_rf = triang.rvs(c=0.5, loc=-res[1], scale=2*res[1]-0.011, size=g, random_state=2020)
                x1_rf = np.zeros(g)
                v1 = np.abs(x1_rf)+0.01
                x0_final = np.concatenate((v0, x0_rf, v1, x1_rf))
                #########
                step2_obj = step2_RF(res[0:d], np.array([sigma_hat]), g, X, Y, Z_long)
                lb_2 = np.zeros(2*k*g) 
                ub_2 = np.concatenate((np.repeat(2*res[0], 2*g), np.repeat(2*res[1], 2*g))) 
                step2_constraint = NonlinearConstraint(step2_obj.step2_constraints_k2, lb_2, ub_2, keep_feasible=True)
                ##########
                equ_rf = np.diag(np.full(2*k*g,1))  
                ########## Create equality constraints
                lb_rf1 = [0]*g + [-res[0]+tol]*g + [0]*g + [-res[1]+tol]*g 
                ub_rf1 = [res[0]-tol]*g + [res[0]-tol]*g + [res[1]-tol]*g + [res[1]-tol]*g
                linear_constraint_rf1 = LinearConstraint(equ_rf, lb_rf1, ub_rf1, keep_feasible=True)
                meth2 = minimize(fun = step2_obj.RF_step2_triangular_k2, x0 = x0_final, method='trust-constr', constraints = [linear_constraint_rf1, step2_constraint], options = {'xtol': 1e-9, 'gtol':1e-9, 'maxiter': 2000})
                rfs = np.concatenate((meth2.x[g:2*g], meth2.x[3*g:4*g]))
            hh = np.concatenate((fxs, [tem1['Success']], [tem1['LK_negative']], [tem1['iteration']], rfs)) 
            containers[t, :] = hh
            print(t)

        tjj = pd.DataFrame(containers)
        name = '_'.join(['Triangular_slope', 'd', str(d), 'N', str(N)])
        final_name = ''.join([name, '.csv'])     
        # In[8]:
        tjj.to_csv(final_name)
