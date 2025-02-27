from scipy.spatial.transform import Rotation as R
import numpy as np

rotation = R.from_euler("xyz", [30, 45, 60], degrees=True)
translation = np.array([1, 2, 3]).reshape(3, 1)

fx = 500
fy = 500
cx = 320
cy = 240

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

dpi = 0.05

delta_p = np.array([2, 1, 0]).reshape(3, 1)
p0 = np.array([500, 801, 1]).reshape(3, 1)
p1 = p0 + delta_p

p0_trans = K @ rotation.as_matrix() @ np.linalg.inv(K) @ p0 + K @ translation * dpi
p1_trans = K @ rotation.as_matrix() @ np.linalg.inv(K) @ p1 + K @ translation * dpi

print(p1_trans)
print(p0_trans)

p0_trans /= p0_trans[2]
p1_trans /= p1_trans[2]

print(p1_trans - p0_trans)


rplane = K @ rotation.as_matrix() @ np.linalg.inv(K)
print(rplane[:2, :2] @ delta_p[:2,:])

print(np.linalg.inv(K))