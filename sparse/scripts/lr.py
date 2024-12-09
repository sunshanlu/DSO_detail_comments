# 验证正则化对优化的影响

import numpy as np
import matplotlib.pyplot as plt

Jo = np.linspace(0, 10, 10)
Ho = np.linspace(0, 10, 10)
x = np.linspace(-10, 10, 1000)

Jo, Ho = np.meshgrid(Jo, Ho)

for Joi, Hoi in zip(Jo, Ho):
    Jni = Jo + 2 * x
    Hni = Hoi + 2
    plt.plot(x, Jni / Hni, label=f"Jo = {Joi}, Ho = {Hoi}")

plt.legend()
plt.show()
