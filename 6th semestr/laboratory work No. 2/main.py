import time
import numpy as np
import matplotlib.pyplot as plt

L = 1
epsilon = 0.0001
h_1 = 0.05
h_2 = 0.05


def u_1(x, y):
    return 2 * x + 3 * y - 5


def f_1(x, y):
    return 0


def u_2(x, y):
    return 2 * x ** 2 + y - 3


def f_2(x, y):
    return -4


def u_3(x, y):
    return 3 * x ** 2 - 2 * y ** 2 - 1


def f_3(x, y):
    return -2


def u_4(x, y):
    return np.exp((np.sin(x)) ** 2 + (np.cos(y)) ** 2)


def f_4(x, y):
    return -1 * np.exp(
        (np.sin(x)) ** 2 + (np.cos(y)) ** 2) * \
           ((np.sin(2 * x)) ** 2 + 2 * np.cos(
               2 * x) + np.sin(2 * y) -
            2 * np.cos(2 * y))


def Cross_Jac(f, h_1, h_2, y0, ep):
    tic = time.perf_counter()
    C_0 = (0.5 * (h_1 ** 2) * (h_2 ** 2)) / (
            h_1 ** 2 + h_2 ** 2)
    C_1 = (0.5 * (h_1 ** 2)) / (h_1 ** 2 + h_2 ** 2)
    C_2 = (0.5 * (h_2 ** 2)) / (h_1 ** 2 + h_2 ** 2)
    Nx = int(L / h_1) + 1
    Ny = int(L / h_2) + 1
    U = np.zeros((Ny, Nx))
    U_m = np.zeros((Ny, Nx))
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)
    # X, Y = np.meshgrid(x, y)
    # Z = u_2(X, Y)

    # Заполняем начальное приближение с учетом нач.условий
    for i in range(0, Ny):
        for j in range(0, Nx):
            U[i][j] = y0(x[j], 0)
    for i in range(0, Nx):  # y = 1
        U[Ny - 1][i] = y0(x[i], L)
    for i in range(0, Ny):  # x = 1
        U[i][Nx - 1] = y0(L, y[i])
    for i in range(0, Ny):  # x = 0
        U[i][0] = y0(0, y[i])

    norm = 1
    k = 0
    for i in range(0, Ny):
        for j in range(0, Nx):
            U_m[i][j] = y0(x[j], 0)
    for i in range(0, Nx):  # y = 1
        U_m[Ny - 1][i] = y0(x[i], L)
    for i in range(0, Ny):  # x = 1
        U_m[i][Nx - 1] = y0(L, y[i])
    for i in range(0, Ny):  # x = 0
        U_m[i][0] = y0(0, y[i])

    while norm > ep:
        for i in range(Nx - 2, 0, -1):
            for j in range(1, Ny - 1):
                U_m[j][i] = C_0 * f(x[i], y[j]) + C_1 * (U[j - 1][i] + U[j + 1][i]) + C_2 * (U[j][i - 1] + U[j][i + 1])
        norm = np.max(np.abs(U_m - U))
        k += 1
        print(k)
        for i in range(0, Ny):
            for j in range(0, Nx):
                U[i][j] = U_m[i][j]

    toc = time.perf_counter()
    tme = round(toc - tic, 4)
    print("Количество итераций = ", k)
    # print("Погрешность с точным решением: ",
    #       np.max(np.abs(Z - U_m)))
    return U_m


def Cross_Jac_with_param(f, h_1, h_2, y0, omega, ep):
    Nx = int(L / h_1) + 1
    Ny = int(L / h_2) + 1
    U = np.zeros((Ny, Nx))
    U_m = np.zeros((Ny, Nx))
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)
    X, Y = np.meshgrid(x, y)
    Z = u_3(X, Y)

    # Заполняем начальное приближение с учетом нач.условий
    for i in list(range(0, Ny)):
        for j in list(range(0, Nx)):
            U[i][j] = y0(x[j], 0)
    for i in range(0, Nx):  # y = 1
        U[Ny - 1][i] = y0(x[i], L)
    for i in range(0, Ny):  # x = 1
        U[i][Nx - 1] = y0(L, y[i])
    for i in range(0, Ny):  # x = 0
        U[i][0] = y0(0, y[i])

    norm = 0
    k = 0
    for i in range(0, Ny):
        for j in range(0, Nx):
            U_m[i][j] = y0(x[j], 0)
    for i in range(0, Nx):  # y = 1
        U_m[Ny - 1][i] = y0(x[i], L)
    for i in range(0, Ny):  # x = 1
        U_m[i][Nx - 1] = y0(L, y[i])
    for i in range(0, Ny):  # x = 0
        U_m[i][0] = y0(0, y[i])

    flag = True
    while flag:
        norm = 0
        for s in reversed(range(1, Ny - 1)):
            for i in reversed(range(1, Nx - 1)):
                U_m[i][s] = (1 - omega) * U[i][s] + (omega / 4) * (
                            U[i - 1][s] + U[i + 1][s] + U[i][s - 1] + U[i][s + 1]) + ((omega * (h_1 ** 2)) / 4) * f(
                    x[i], y[s])
                if abs(U_m[i][s] - U[i][s]) > norm: norm = abs(U_m[i][s] - U[i][s])
        if norm < ep:
            flag = False
        k += 1
        print(k)
        U = U_m.copy()
    print("количество итераций = ", k)
    print("Погрешность с точным решением: ",
          np.max(np.abs(U_m - Z)))
    return U_m


def Table(name, func, h_1, h_2, y0, ep):
    with open(name, 'w') as f:
        f.write('& ' + 'h_{0} & ' + 'h_{0}/2 & '
                + 'h_{0}/4 & ' + 'h_{0}/8 &' + 'h_{0}/16 '
                + ' \\\ ' + '\hline' + '\n')
        for i in range(0, 5):
            epsil = ep / (10 ** i)
            for j in range(0, 5):
                h1 = h_1 / (2 ** (j - 1))
                h2 = h_2 / (2 ** (j - 1))
                tm, it = Cross_Jac(func, h1, h2,
                                   y0,
                                   epsil)
                if j == 4:
                    f.write(
                        ' ' + "{:1.4G}".format(
                            it) + ' ' + "{:1.4G}".format(
                            tm))
                    f.write(' \\\ ' + '\hline')
                    f.write('\n')
                    print("Итераций, time", it, " ", tm)
                elif j == 0:
                    f.write('\\' + 'varepsilon_{0}/{'
                            + str(10 ** i) + '} & ')
                    print(2 ** i)
                else:
                    f.write("{:1.4G}".format(
                        it) + ' ' + "{:1.4G}".format(tm))
                    f.write(' & ')
                    print("Итераций, time ", it, " ", tm)


#
# omn = h_2**2 / 4
Um = Cross_Jac(f_3, h_1, h_2, u_3, epsilon)

# Table("Табличка.txt", f_3, h_1, h_2, u_3, epsilon)

Nx = int(L / h_1) + 1
Ny = int(L / h_2) + 1

fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y)
# Z = u_2(X, Y)
#
plt.xlabel("x")
plt.ylabel("y")
ax.plot_surface(X, Y, Um, color='blue')
# ax.plot_surface(X, Y, Z, color='red')

plt.show()
