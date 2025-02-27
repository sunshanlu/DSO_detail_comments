import numpy as np
import math

A = np.array([100000, 100000, 0, 1e-6]).reshape(2, 2)
A_inv = np.linalg.inv(A)

x_true = np.array([1, 0]).reshape(2, 1)
b_true = A @ x_true
b_intp = b_true + np.array([0, 1e-5]).reshape(2, 1)
x_intp = A_inv @ b_intp


D = np.array([10000000, 1e-6])
W = np.diag(1.0 / np.sqrt(D + 10))
x_trans = W @ (np.linalg.inv(W @ A @ W) @ (W @ b_intp))

print(x_true)
print(x_intp)
print(x_trans)

print(math.log(np.linalg.cond(A)))
print(math.log(np.linalg.cond(W @ A @ W)))
