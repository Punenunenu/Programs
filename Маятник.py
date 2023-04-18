import numpy as np
from numpy import cos
from numpy import pi as π
import matplotlib.pyplot as plt

g = 9.81
m = 1
k = 0.981
x0 = m*g/k 
ω = (k/m)**0.5
xm = 1
φ0 = 0
t_distr = 2.5
fx = lambda t, A = xm, w0 =ω, fi0 = φ0: A * cos(w0*t + fi0)
start_time = 0
end_time = 10
dt = 0.001
time = list(np.arange(start_time, end_time, dt))
mas_x = []

for t in time:
    if t < (start_time + t_distr):
        mas_x.append(x0)
    else:
        mas_x.append(x0 + fx(t))

fg, ax = plt.subplots(figsize=(7.5, 3.5), layout='constrained')
ax.plot(time, mas_x, label=f'x')
ax.set_xlabel('Время, с')
ax.set_ylabel('X, м')
ax.set_title("График горманических колебаний пружиного маятника")
ax.grid()
plt.show()
