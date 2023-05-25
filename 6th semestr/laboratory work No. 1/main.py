import numpy as np
from matplotlib import pyplot as plt

L = 1
T = L
Nl = 60
Nt = 1001

def y1(x, t):
    return 3 * x**2 + 2 * t**2 + 6
def f1(x, t):
    return -2
def y01(x):
    return 3 * x**2 + 6
def g01(t):
    return 2 * t**2 + 6
def gl1(t):
    return 2 * t**2 + 9
def v1(x):
    return 0

def Krest(f, Nl, Nt, y0, g0, gl, v):
    U = np.zeros((Nl, Nt))
    x = np.linspace(0, L, Nl)
    t = np.linspace(0, T, Nt)
    x1 = L/(Nl - 1)
    t1 = T/(Nt - 1)
    for i in range(0, Nl):
        U[i][0] = y0(x[i])
    for j in range(0, Nt):
        U[0][j] = g0(t[j])
    for j in range(0, Nt):
        U[Nl - 1][j] = gl(t[j])
    for i in range(1, Nl - 1):
        U[i][1] = v(x[i]) * t1 + (t1**2/(2*x1**2)) * (U[i+1][0] - 2 * U[i][0] + U[i-1][0]) + (t1**2)/2 * f(x[i], t[0]) + U[i][0]
    for i in range(1, Nl - 1):
        for j in range(2, Nt - 1):
            U[i][j+1] = (t1**2/x1**2) * (U[i+1][j] - 2*U[i][j] + U[i-1][j]) + t1**2 * f(x[i], t[j]) + 2*U[i][j] - U[i][j-1]
    return U
UU = Krest(f1, Nl, Nt, y01, g01, gl1, v1)

x = np.linspace(0, L, Nl)
t = np.linspace(0, T, Nt)
st = 5
tt = t[st]

print(tt)

sol_x = np.zeros(Nl)
uh = []

for i in range(0, Nl):
    sol_x[i] = y1(x[i], tt)
    uh.append(UU[i][st])
print(uh)
plt.plot(x, sol_x,'r', label='точное', linewidth=2)
plt.plot(x, uh,'b', label='полученное', linewidth=2)
# plt.xlim(0,0.1)
# plt.ylim(-10,10)
plt.grid()
plt.legend()
plt.title('')
plt.show()