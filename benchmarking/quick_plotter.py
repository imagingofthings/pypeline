import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
ps_total_runtime =     [16.67, 51.98, 96.34,  186.32,  348.94, 870.49, 1608]
ss_total_runtime =     [27.,   69.42, 174.83, 359.47,  760.32, 1671.02, 3351]
kernel_regen_runtime = [21.19, 21.84, 22.04,  22.30,   39.37,  100., 172.98]
nsteps =               [5,     20,    50,     100,     200,    500,   1000]

fig, ax = plt.subplots()
ax.plot(nsteps, ps_total_runtime,     'b' ,  label = "Fast Periodic Synthesis")
ax.plot(nsteps, kernel_regen_runtime, 'b--', label = "Kernel Regen Time")
ax.plot(nsteps, ss_total_runtime,     'r',   label = "Standard Periodic Synthesis")


ax.set(xlabel='total timesteps', ylabel='runtime (s)',
       title='')
ax.grid()

legend = ax.legend(loc='upper center', fontsize='x-large')

fig.savefig("test.png")
plt.show()