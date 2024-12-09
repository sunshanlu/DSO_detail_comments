import matplotlib.pyplot as plt
import numpy as np

fov = lambda r, omega: (1 / omega) * np.arctan(2 * r * np.tan(omega / 2)) / r

# r固定，看omega对fov的ratio的影响
r_num = 10
omega = np.linspace(0.1, 1, 1000)
r = [x * np.ones_like(omega) for x in np.linspace(0.1, 1, r_num)]


for r_i in r:
    ratio = fov(r_i, omega)
    plt.plot(omega, ratio, label=f"r = {r_i[0]}")

plt.legend()
plt.show()

# 结论，在omega固定的情况下，r越大，畸变的比例会变大，并且随着r变大，畸变比例变大的越快，感觉这种是否符合雨眼相机的去畸变描述
