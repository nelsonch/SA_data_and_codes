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
from laplace_sadd import laplace_sda, equlity_constraints_laplace, get_near_psd, step2_RF, RF_step2_laplace_simple 
# In[2]:
# In[2]:
M = 10 #### number of repeats for the number. 
repeats = 5  #### # of repeats for initial values. 
k = 1 ##### dimension of Z
g = 20 ##### # of groups
tol = 1e-05
# In[2]:
d_matrix = np.array([8])
N_matrix = np.array([100, 200, 500])


for d in d_matrix:
    for N in N_matrix:
        n = d
        if d == 8:
            true = np.array([1.5, 1.5, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 0.3, 0.1])
            RF_index = np.array(['Y', 'N', 'N', 'N', 'N', 'N', 'N', 'N'])
        else:
            true = np.array([1.5, 1.5, 1.0, 1.5, 1.5, 0.3, 0.1])
            RF_index = np.array(['Y', 'N', 'N', 'N', 'N'])

        containers = np.zeros((M, len(true) + 3 + k*g + k*g + 2))
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
        sigma = true[n] #### the following 1 is sigma 

        # In[5]:
        np.random.seed(2019)
        gamma = np.random.laplace(0, true[-1], size = g)


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
            equ = np.zeros((d+1+k+N, d+1+k+N))
            np.fill_diagonal(equ, 1)
            lb = [tol]*d + [-100] + [tol] + [-100]*N 
            ub = [100]*d + [100] + [100] + [100]*N
            sign_constraint = LinearConstraint(equ, lb, ub, keep_feasible=True)

            ###### uniform K' = Y constraints
            eq_cons = equlity_constraints_laplace(tol, RF_index, X, Y, Z)
            lb_e = Y; ub_e = Y
            equality_constraint = NonlinearConstraint(eq_cons.gradient_K_fun_laplace, lb_e, ub_e)

            ########
            ####### inequality constraints 
            #eq_cons = equlity_constraints_laplace(tol, RF_index, X, Y, Z)
            lb_ie = np.zeros(k*N); ub_ie = np.ones(k*N)-tol
            inequality_constraint = NonlinearConstraint(eq_cons.ineq_constraints_laplace, lb_ie, ub_ie, keep_feasible=False)

            ########
            iniseed = np.array([2017, 1989, 1990, 1991, 2077])
            initials = np.zeros((len(iniseed), d+1+k+N))
            for j in range(len(iniseed)):
                if d==5:
                    tem0 = uniform.rvs(loc=true[0], scale=0.2, size=1, random_state=int(iniseed[j]))
                    tem1 = uniform.rvs(loc=true[1], scale=0.2, size=1, random_state=int(iniseed[j]))
                    tem2 = uniform.rvs(loc=true[2], scale=0.2, size=1, random_state=int(iniseed[j]))
                    tem3 = uniform.rvs(loc=true[3], scale=0.2, size=1, random_state=int(iniseed[j]))
                    tem4 = uniform.rvs(loc=true[4], scale=0.2, size=1, random_state=int(iniseed[j]))
                    baba = np.array([tem0, tem1, tem2, tem3, tem4]).ravel()
                else:
                    tem0 = uniform.rvs(loc=true[0], scale=0.2, size=1, random_state=int(iniseed[j]))
                    tem1 = uniform.rvs(loc=true[1], scale=0.2, size=1, random_state=int(iniseed[j]))
                    tem2 = uniform.rvs(loc=true[2], scale=0.2, size=1, random_state=int(iniseed[j]))
                    tem3 = uniform.rvs(loc=true[3], scale=0.2, size=1, random_state=int(iniseed[j]))
                    tem4 = uniform.rvs(loc=true[4], scale=0.2, size=1, random_state=int(iniseed[j]))
                    tem5 = uniform.rvs(loc=true[5], scale=0.2, size=1, random_state=int(iniseed[j]))
                    tem6 = uniform.rvs(loc=true[6], scale=0.2, size=1, random_state=int(iniseed[j]))
                    tem7 = uniform.rvs(loc=true[7], scale=0.2, size=1, random_state=int(iniseed[j]))
                    #tem8 = uniform.rvs(loc=true[8], scale=0.2, size=1, random_state=int(iniseed[j]))
                    baba = np.array([tem0, tem1, tem2, tem3, tem4, tem5, tem6, tem7]).ravel()

                #tem1 = uniform.rvs(loc=1.0, scale=2.0, size=d-1, random_state=int(iniseed[j]))
                tem9 = uniform.rvs(loc=true[n], scale=0.02, size=1, random_state=int(iniseed[j]))
                tem10 = uniform.rvs(loc=true[-1], scale=0.02, size=1, random_state=int(iniseed[j]))
                #tem1 = true[0:d] + uniform.rvs(loc=-0.1, scale=0.2, size=d, random_state=iniseed[j])
                #tem2 = true[d:(d+1)] + uniform.rvs(loc=-0.05, scale=0.10, size=1, random_state=iniseed[j])
                t_start = uniform.rvs(loc=-0.5, scale=1.0, size=N, random_state=iniseed[j])
                initials[j, :] = np.concatenate((baba, tem9, tem10, t_start))


            ####### get uniform object 
            #uni_obj = triangular_sda(tol, g, RF_index, X, Y, Z)
            uni_obj = laplace_sda(tol, g, RF_index, X, Y, Z)  
            #uni_obj.LK_negative(initials[0, :])

            chacha = np.inf 
            x_final = None 
            success = False
            ite_grand = 0
            for j in range(len(iniseed)):
                x0_t = initials[j, :].ravel()
                meth1 = minimize(fun = uni_obj.LK_negative, x0 = x0_t, method='trust-constr', constraints = [sign_constraint, equality_constraint, inequality_constraint], options={'maxiter': 2000})
                if meth1.success:
                    success = True
                    if ((meth1.fun <= chacha) & (meth1.fun > -5000)):
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
                print(meth1.x[0:(d+1+k)])
            # In[14]:
            tem1 = {'results': x_final, 'iteration': ite_grand, 'LK_negative': chacha, 'Success': success}
            #tem1 = {'results': x_final, 'iteration': ite_grand, 'LK_negative': chacha, 'Success': success}
            # In[25]:
            print(tem1)
            res = x_final
            fxs = res[0:(d+1+k)]
            res = tem1['results']
            if res is None:
                fxs = np.zeros(d+1+k)
                rfs1 = np.zeros(g*k)
                rfs2 = rfs1
                rmse1 = 100
                rmse2 = 100
            else:
                fxs = res[0:(d+1+k)]
                sigma_hat = fxs[-2]
                #print(sigma_hat)
                b_v = fxs[-1]
                #print(b_v)
                ##########
                step2_obj = step2_RF(res[0:d], np.array([sigma_hat]), np.array([b_v]), g, X, Y, Z_long)
                lb_2 = np.zeros(2*g)
                ub_2 = np.repeat(2*100, 2*g)
                step2_constraint = NonlinearConstraint(step2_obj.step2_constraints, lb_2, ub_2, keep_feasible=True)
                #########
                x0_rf = np.zeros(g)
                v0 = np.abs(x0_rf)+0.001
                x0_final = np.concatenate((v0, x0_rf))
                ##############
                meth2 = minimize(fun = step2_obj.RF_step2_laplace, x0 = x0_final, method='trust-constr', constraints = [step2_constraint], options = {'xtol': 1e-10, 'gtol':1e-10, 'maxiter': 2000})
                rfs1 = meth2.x[g:2*g]    
                ##########
                meth3 = minimize(fun = RF_step2_laplace_simple, x0 = np.zeros(g), args = (res[0:d], np.array([sigma_hat]), g, X, Y, Z_long), method='trust-constr', options = {'xtol': 1e-10, 'gtol':1e-10, 'maxiter': 2000})
                rfs2 = meth3.x
                rmse1 = np.sqrt(np.sum((meth2.x[g:2*g] - gamma)**2)/g)
                rmse2 = np.sqrt(np.sum((meth3.x - gamma)**2)/g) 
                print('RF rmse1 of old')
                print(rmse1)
                print('RF rmse2 of distance')
                print(rmse2)
                #rfs = meth2.x[g:2*g]
            hh = np.concatenate((fxs, [tem1['Success']], [tem1['LK_negative']], [tem1['iteration']], rfs1, rfs2, [rmse1], [rmse2])) 
            containers[t, :] = hh
            print(t)
        
        tjj = pd.DataFrame(containers)
        name = '_'.join(['Laplace_intercept', 'd', str(d), 'N', str(N)])
        final_name = ''.join([name, '.csv'])     
        # In[8]:
        tjj.to_csv(final_name)
