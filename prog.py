import numpy as np
import pprint
import scipy
import scipy.linalg

# def gaussElim(A): 
#     # ”””
#     # Perform Gaussian Elimination. 
#     # Warning the code overwrites A.

#     # Note: Numpy like C derived languages use indices starting at 0, since they are all interpreted as pointer offsets .
#     # Note that all the math we are doing uses indices starting at 1, so you need to translate those. 
#     # ”””
#     A=A.astype(float)
#     nrows = A.shape[0] 
#     ncols = A.shape[1]
    
#     # L = np.diag(np.full(nrows,1))
#     # L = np.zeros_like(A)
#     # np.fill_diagonal(L, 1.)
#     # print(type(L), ' ', L.shape, " ", L.dtype)

#     for k in range(ncols-1): # Iterate over all columns 
#         for i in range(k+1, nrows): # Walk down the column
#             lik = A[i, k] / A[k, k]
#             # A[i, k]=0
#             A[i,k] = lik

#             # Do row operation
#             for j in range(k+1, ncols): 
#                 A[i, j] -= lik * A[k, j]
#     return A

def swap(p, row1, row2):
    if row1 == row2: return
    p[row1],p[row2]=p[row2],p[row1]
    
def gaussElim(A): 
    # ”””
    # Perform Gaussian Elimination. 
    # Warning the code overwrites A.

    # Note: Numpy like C derived languages use indices starting at 0, since they are all interpreted as pointer offsets .
    # Note that all the math we are doing uses indices starting at 1, so you need to translate those. 
    # ”””
    
    A=A.astype(float)
    nrows = A.shape[0] 
    ncols = A.shape[1]
    
    P = np.arange(nrows)
    # L = np.zeros_like(A)
    # np.fill_diagonal(L, 1.)
    # print(type(L), ' ', L.shape, " ", L.dtype)

    for k in range(ncols-1): # Iterate over all columns 
        max_row = np.argmax(np.abs(A[k:, k]))+k
        swap(P, k, max_row)
        for i in range(k+1, nrows): # Walk down the column
            lik = A[P[i], k] / A[P[k], k]
            A[P[i],k] = lik

            # Do row operation
            for j in range(k+1, ncols): 
                A[P[i], j] -= lik * A[P[k], j]
    return A, P

# Example
# A = np.array([[2, 1, 3, 1], 
#               [4, 4, 7, 1],
#               [2, 5, 9, 3]])

# A = gaussElim(np.array([[1,2],
#              [3,4]]))

# A = gaussElim(np.array([[4,1,0],
#                         [1,4,1],
#                         [0,1,4]]))

# U = np.triu(A)
# L = np.tril(A)
# np.fill_diagonal(L, 1.)

# print(L)
# print(U)
# print(np.dot(L,U))

# A =  np.array([[1,2],
#                [3,4]]) 
# P,L,U = scipy.linalg.lu(A) 
# pprint.pprint(P)
# pprint.pprint(L)
# pprint.pprint(U)

# print(np.array_equal(np.dot(P,A),np.dot(L,U)))

# A = np.array([[1, 1, 1], 
#               [1, 1, 2],
#               [1, 2, 2]])

# A = np.array([[1,2], [3,4]])

# A = np.array([[1, 1, 1], 
#               [1, 1, 2],
#               [1, 2, 2]])

# P, L, U = scipy.linalg.lu(A)

# print(np.array_equal(np.dot(P,A), np.dot(L,U)))

# print(np.array_equal(A[P,:], np.dot(L,U)))

# print(np.diag(np.full(A.shape[0],1)))

# print(gaussElim(A))

# A,P  = gaussElim(A)

# def horner(coeffs, n, x):
#     if n == 0: return coeffs[n]
#     return coeffs[n-1]*x**(n-1) + horner(coeffs, n-1, x)

horner = lambda coeffs, n, x: coeffs[n] if n == 0 else coeffs[n-1]*x**(n-1) + horner(coeffs, n-1, x)

# m = 100
# xi = np.linspace(-1,1,m+1)
# fi = 1/(1+25*xi**2)
# fi = np.sin(xi)

# n = len(xi)
# A = np.array([[x**j for j in range(n)] for x in xi])
# B = np.array(fi)

# print(scipy.linalg.solve(A, B))

# import matplotlib.pyplot as plt

# x = np.linspace(-10,10,1000)
# plt.plot(x, horner(scipy.linalg.solve(A,B), n, x))
# plt.show()

# Example 1: Consider the decay of coefficients of f(x) = sign(x)
N = 1001
y = np.linspace(-1, 1, N)
g = 1/(1+25*y**2)

norms = []
ns = 2*np.unique(np.round(2**np.arange(0, 7, 0.3))) - 1
print(ns)

import matplotlib.pyplot as plt

for n in ns:
    x = np.linspace(-1,1,int(n))
    f = 1/(1+25*x**2)
    
    A = np.array([[_**j for j in range(len(x))] for _ in x])
    B = np.array(f)

    # Compute ||f - pn||_inf
    h = horner(scipy.linalg.solve(A,B), len(x), y)
    norms.append(np.max(np.abs(h - g)))

plt.loglog(ns, norms, marker='o')
plt.loglog(ns, np.power(ns, -1))
plt.legend(['$||p_n - g||_{\infty}$', '$1/n$'], loc='best')
plt.title('Plot of norm of error of interpolant against number of data points')
plt.xlabel('$n$')
# plt.ylabel('$||p_n - g||_{\infty}$')
plt.ylabel('y')
plt.savefig('plot1.png')
plt.show()