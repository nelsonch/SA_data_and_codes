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
#from functions_saddle_uniform_slope import *
from uniform_sadd import uniform_sda, equlity_constraints_uniform, RF_step2_uniform, get_near_psd
#get_ipython().run_line_magic('config', 'Completer.use_jedi = False')

# In[ ]:

dat = pd.read_csv('try668.csv')
df = dat.sort_values('store_clus')
ss = df.groupby('store_clus')['logistic_qty'].size().ravel()

d=n=2
g=len(ss)
group_index = np.repeat(np.linspace(1, g, g), ss).astype(int)
unique_levels = len(np.unique(group_index))

#### get the dummy matrix
pd_group_index = pd.DataFrame(group_index, columns = ['group_index'])
dummy_matrix = pd.get_dummies(pd_group_index['group_index'].astype('category',copy=False), prefix='x0')
Y = df['logistic_qty'].values
#len(Y)


# In[11]:
X = df[['intcp', 'disc']]
Z = X
#X.head(4)

N = X.shape[0]; d=n=2; k=2
m_vector = np.ones(k)[:, np.newaxis]


RF_index = np.array(['Y', 'Y'])
RF_true = np.where(RF_index == 'Y')[0].astype(int)
X_df = pd.DataFrame(X)
X_df_subset = X_df.iloc[:, RF_true]


Z1 = pd.DataFrame()
for j in range(X_df_subset.shape[1]):
    tem = pd.DataFrame(X_df_subset.iloc[:, j].values[:, np.newaxis]*dummy_matrix.values)
    Z1 = pd.concat([Z1, tem], axis=1)
Z_long = Z1.values

##############################
tol=1e-5
equ = np.zeros((d+1+N, d+1+N))
np.fill_diagonal(equ, 1)
lb = [tol, tol, tol] + [-100]*N
ub = [2, 0.5, 2] + [100]*N
sign_constraint = LinearConstraint(equ, lb, ub, keep_feasible=True)


tol = 1e-05
eq_cons = equlity_constraints_uniform(tol, RF_index, X.values, Y, Z.values)
lb_e = Y; ub_e = Y
equality_constraint = NonlinearConstraint(eq_cons.gradient_K_fun_uniform, lb_e, ub_e)


#ot = np.array([251, 10.5, 17.7, 4.2, 25.56])
iniseed = np.array([2017, 1989, 1990, 1991, 2077])
repeats = 4
initals = np.zeros((repeats, len(lb)))
ii = 0
for ii in range(repeats):
    o0 = np.array([0.30]) 
    o01 = np.array([tol]) 
    o1 = np.array([1.39])
    if ii < 2:
        t_start = uniform.rvs(loc=-1, scale=2, size=N, random_state=iniseed[ii])
    else:
        t_start = uniform.rvs(loc=-5, scale=10, size=N, random_state=iniseed[ii])
    initals[ii, :] = np.concatenate((o0, o01, o1, t_start))
    #initals[ii, :] = np.array([251.41, 10.47, 25.57])
initial_final = initals
m_vector = np.ones(k)[:, np.newaxis]


uni_obj = uniform_sda(tol, g, RF_index, X.values, Y, Z.values) 

chacha = np.inf 
x_final = None 
success = False
ite_grand = 0
for j in range(repeats):
    x0_t = initial_final[j, :].ravel()
    meth1 = minimize(fun = uni_obj.LK_negative, x0 = x0_t, method='trust-constr', constraints = [sign_constraint, equality_constraint], options={'maxiter': 1000})
    if meth1.success:
        success = True
        if ((meth1.fun <= chacha) & (meth1.fun > -1000000)):
            if len(np.where((meth1.x[0:d] > 0) == False)[0]) < 1:
                #x_final = np.concatenate((meth1.x[0:d], np.array([meth1.x[d]]), meth1.x[(d+1):]))
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


# In[ ]:
tem1 = {'results': x_final, 'iteration': ite_grand, 'LK_negative': chacha, 'Success': success}
print(tem1)

res = tem1['results']
beta_v = res[0:k]
sigma = res[k:(k+1)]

x0_rf = uniform.rvs(loc=-0.1, scale=0.2, size=k*g, random_state=2021)
##### contraints for RF
equ_rf = np.diag(np.full(g*k, 1))  
########## Create equality constraints
#lb_rf = [-np.inf]*g + [-np.inf]*g 
#ub_rf = [np.inf]*g + [np.inf]*g 
lb_rf = [-100]*g + [-res[1]]*g 
ub_rf = [100]*g + [res[1]]*g 
#ub_rf = [np.inf, np.inf, -1e-05, -1e-05]
linear_constraint_rf = LinearConstraint(equ_rf, lb_rf, ub_rf)


def RF_step2(gamma, beta, sigma, X = None, Z_long = None, Y = None):
    E = Y.ravel() - (X@beta[:, np.newaxis]).ravel() - (Z_long@gamma).ravel()
    tem1 = np.sum(E**2/sigma**2)
    #######
    #lambda_v_matrix = np.diag(np.repeat(1/np.sqrt(2)/lambda_v, g))
    #tem2_1 = lambda_v_matrix@gamma[:, np.newaxis]
    #tem2 = np.sum((gamma[np.newaxis, :]@tem2_1).ravel())
    #return tem1 + tem2 
    return tem1

################# 2nd Step optimization Method 1
#### get the results from optimization 1
########### optimization 
meth2 = minimize(fun = RF_step2, x0 = x0_rf, 
                 args = (beta_v, sigma, X.values, Z_long, Y), 
                 method='trust-constr', 
                 constraints = [linear_constraint_rf], options = {'xtol': 1e-8, 'gtol':1e-8, 'maxiter': 2000})
#end_time = time.time()

print(meth2.success)
print(meth2.x)


rf0 = meth2.x[0:g]
rf1 = meth2.x[g:g*k]

#len(rf2)
overall1 = res[0] + rf0
overall2 = res[1] + rf1

print(overall1.round(5))

print(overall2.round(5))


# In[ ]:

th = np.concatenate((overall1[:, np.newaxis], overall2[:, np.newaxis]), 1)

#pd.DataFrame(th).to_csv('Uniform.csv')

gamma = meth2.x
E = (X.values@beta_v[:, np.newaxis]).ravel() + (Z_long@gamma).ravel()
baba = np.sqrt(np.sum((Y-E)**2)/N)
print(baba)

