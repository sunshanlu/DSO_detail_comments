# 针对真实情况 H J

import numpy as np

# A代表系数矩阵
A = np.zeros((2, 2), np.float32)
A[0, 0] = 2
A[1, 1] = 5

# 代表优化初始值
x0 = np.array([1, 2], np.float32)

# 定义针对x参数的雅可比矩阵
get_jacobian = lambda x: np.array([2 * x[0], 2 * x[1]]).reshape(1, 2)
get_hessian = lambda x: np.array([[2, 0], [0, 2]], np.float32)

J = get_jacobian(x0)
H = get_hessian(x0)

H_n = A @ get_hessian(x0) @ A
J_n = J @ A

delta_x = -np.linalg.inv(H) @ J.T
delta_x_n = -A @ np.linalg.inv(H_n) @ J_n.T
print(delta_x)
print(delta_x_n)