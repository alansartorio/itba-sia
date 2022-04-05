from selection import temperature
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 100, 100)

for k in [0.001, 0.01, 0.1, 1]:
    plt.plot(x, [temperature(v, k, T0=10000) for v in x], label=f'k={k}')

plt.ylim(0)
plt.xlabel('Generations')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('temperature_function.png', dpi=300)
plt.show()
