#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import scipy.sparse as sps


# In[8]:


def thomas(A, b):
    
    # c_1' = c_1 / b_1
    A[0,1] = A[0,1] / A[0,0]
    
    # d_1' = d_1 / b_1
    b[0] = b[0] / A[0,0]
    
    # b_1 = 1
    A[0,0] = 1.
   
    # Get dimmension
    n = A.shape[0]
    
    for i in range(1, n - 1):
        # d_n'=(d_n - a_n * d_{n-1}')/ (b_n - a_n * c_{n-1}')
        b[i] = (b[i] - A[i, i-1] * b[i-1]) / (A[i,i] - A [i, i-1] * A[i-1,i])
        
        # c_n' = c_n /(b_n - a_n * c_{n-1}')
        A[i,i+1] = A[i,i+1] / (A[i, i] - A[i, i-1] * A[i-1,i])
        
        # b_n = 1
        A[i,i] = 1.
        
        # a_n = 0
        A[i,i-1] = 0.
       
    
    b[n-1] = (b[n-1] - A[n-1,n-2] * b[n-2]) / (A[n-1,n-1] - A[n-1,n-2] * A[n-2,n-1])
    
    A[n-1,n-1] = 1.
    
    A[n-1,n-2] = 0
    
    x_soln = np.zeros_like(b)
    x_soln[n - 1] = b[n - 1]
    
    for i in range(n -2, -1, -1):
        # x_n = d_n - c_n' * d_{n+1}
        x_soln[i] = b[i] - A[i,i+1] * x_soln[i+1]
        
    return x_soln

