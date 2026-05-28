import numpy as np
import matplotlib.pyplot as plt

r0_layers = np.array([0.0506, 0.3895, 0.2015])  # in meters

r0_total = (np.sum(r0_layers**(-5/3)))**(-3/5)

print(r0_total)
quit()

N = 1000
D = 4
size = 1000

r0_to_var = lambda r0: 1.0299 * (D/r0) ** (5/3)
var_to_r0 = lambda var: D * (1.0299 / var) ** (3/5)

r0_int = []
r0_sum = []
r0_exp = []

var_int = []
var_sum = []
var_exp = []


for _ in range(N):
    r0 = np.random.normal(0.4, 0.1, 3)
    # r0_t1 = (r0[0] ** (-5/3) + r0[1] ** (-5/3) + r0[2] ** (-5/3)) ** (-3/5)
    r0_t1 = np.sum(r0**(-5/3))**(-3/5)

    var_int.append(r0_to_var(r0_t1))

    phi_a = np.random.normal(0, np.sqrt(r0_to_var(r0[0])), size=size)
    phi_b = np.random.normal(0, np.sqrt(r0_to_var(r0[1])), size=size)
    phi_c = np.random.normal(0, np.sqrt(r0_to_var(r0[2])), size=size)

    phi_t = phi_a + phi_b + phi_c

    r0_t2 = var_to_r0(np.var(phi_t))
    var_sum.append(np.var(phi_t))

    r0_a = var_to_r0(np.var(phi_a))
    r0_b = var_to_r0(np.var(phi_b))
    r0_c = var_to_r0(np.var(phi_c))

    r0_t3 = (r0_a ** (-5/3) + r0_b ** (-5/3) + r0_c ** (-5/3)) ** (-3/5)
    var_exp.append(r0_to_var(r0_t3))

    r0_int.append(r0_t1)
    r0_sum.append(r0_t2)
    r0_exp.append(r0_t3)

plt.hist(r0_int, bins=50, density=True, alpha=0.5)
plt.figure()
plt.hist(r0_sum, bins=50, density=True, alpha=0.5)
plt.figure()
plt.hist(r0_exp, bins=50, density=True, alpha=0.5)
plt.figure()
plt.hist(np.array(r0_sum) - np.array(r0_exp), bins=50, density=True, alpha=0.5)
plt.figure()
plt.hist(np.array(r0_sum) - np.array(r0_int), bins=50, density=True, alpha=0.5)
plt.figure()
plt.hist(np.array(r0_int) - np.array(r0_exp), bins=50, density=True, alpha=0.5)

print(np.mean(r0_sum))
print(np.mean(r0_exp))
print(np.mean(r0_int))

print(np.mean(var_sum))
print(np.mean(var_exp))
print(np.mean(var_int))

plt.show()