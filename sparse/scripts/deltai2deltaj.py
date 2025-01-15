import numpy as np


def rotation_matrix_euler(roll, pitch, yaw):
    """
    构建基于欧拉角的旋转矩阵，旋转顺序为Z-Y-X（ZYX顺序）。

    Args:
        roll (float): 滚转角（弧度）
        pitch (float): 俯仰角（弧度）
        yaw (float): 偏航角（弧度）

    Returns:
        numpy.ndarray: 3x3旋转矩阵
    """
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cos_roll, -sin_roll], [0, sin_roll, cos_roll]])

    Ry = np.array([[cos_pitch, 0, sin_pitch], [0, 1, 0], [-sin_pitch, 0, cos_pitch]])

    Rz = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def camera_calib_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def project_points(point: np.ndarray, dpi, R, t, K):
    pj = K @ R @ np.linalg.inv(K) @ point.reshape(3, 1) + K @ t * dpi
    pj /= pj[2]
    return pj


def project_points_delta(delta_p: np.ndarray, R, K):
    KRK_inv = K @ R @ np.linalg.inv(K)
    Rplane = KRK_inv[:2, :2]
    return Rplane @ delta_p


roll = np.radians(30)  # 滚转角30度
pitch = np.radians(45)  # 俯仰角45度
yaw = np.radians(60)  # 偏航角60度

# R = rotation_matrix_euler(roll, pitch, yaw)
R = np.eye(3)
t = np.array([1, 1, 1]).reshape(3, 1)
K = camera_calib_matrix(500, 500, 320, 240)

delta_p = np.array([-1, 1, 0]).reshape(3, 1)
point0 = np.array([50, 35, 1]).reshape(3, 1)
point1 = point0 + delta_p

pj0 = project_points(point0, 0.5, R, t, K)
pj1 = project_points(point1, 0.5, R, t, K)
delta_pj = project_points_delta(delta_p[:2, :], R, K)

print(f"正常投影的方式：delta_pj = {(pj1 - pj0).reshape(3)}")
print(f"使用DSO的方式：delta_pj = {delta_pj.reshape(2)}")
