import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import time

h_x = 0.1
h_y = 0.1
Nx = 10
Ny = 10
Pe = 1
# sopt.newton_krylov()

# ------------------------
# Тест №1 - параболоид
# test = 1
# Тест №2 - квадратично-линейная функция
test = 2
# Тест №3 - Индивидульаная функция
# test = 3
# ------------------------


def u_func(i, j):
    x = i * h_x
    y = j * h_y
    match test:
        case 1:
            return 1 - ((x - 1) ** 2) - ((y - 2) ** 2)
        case 2:
            return 0.5 * x ** 2 - 0.5 * y
        case 3:
            return np.sin(x ** 2) + 2 * np.cos(2 * y ** 2)


def g_func(i, j):
    x = i * h_x
    y = j * h_y
    match test:
        case 1:
            return 0
        case 2:
            return np.sin(x * y)
        case 3:
            return x ** 2 - y


def f_func(i, j):
    x = i * h_x
    y = j * h_y
    match test:
        case 1:
            return 4
        case 2:
            return -1 + x * y - 2 * x
        case 3:
            return 4 * x ** 2 * np.sin(x ** 2) - 2 * np.cos(x ** 2) + 8 * np.sin(2 * y ** 2) + 32 * y ** 2 * np.cos(
                2 * y ** 2)


def v1_func(i, j):
    x = i * h_x
    y = j * h_y
    match test:
        case 1:
            return 0
        case 2:
            return y - 1
        case 3:
            return 4 * y * np.sin(2 * y ** 2)


def v2_func(i, j):
    x = i * h_x
    y = j * h_y
    match test:
        case 1:
            return 0
        case 2:
            return 2 * x
        case 3:
            return x * np.cos(x ** 2)


def solve_of_task(U):
    trsol = np.zeros((Nx + 1, Ny + 1))
    for i in range(1, Nx):
        for j in range(1, Ny):
            trsol[i][j] = U(i, j)
    return trsol


def givens(A, N):
    for l in range(N - 1):
        for i in range(N - 1, 0 + l, -1):
            j = i - 1
            if A[i][l] != 0:
                alem = A[j][l]
                belem = A[i][l]
                if np.abs(belem) > np.abs(alem):
                    tau = alem / belem
                    S = 1 / np.sqrt(1 + tau ** 2)
                    C = S * tau
                else:
                    tau = belem / alem
                    C = 1 / np.sqrt(1 + tau ** 2)
                    S = C * tau
                A[i], A[j] = A[i] * C - A[j] * S, A[j] * C + A[i] * S
    return A


def Gauss_back_step(A, B, N):
    sol = np.zeros(N)
    for i in range(N - 1, -1, -1):
        s = 0
        if i == N - 1:
            sol[i] = B[i] / A[i][i]
        else:
            for j in range(i + 1, N, 1):
                s += A[i][j] * sol[j]
            sol[i] = (B[i] - s) / A[i][i]
    return sol


# произведение матрицы системы A на произвольный массив p
def multA(p):
    Ap = np.copy(p)
    for i in range(1, Ap.shape[0] - 1):
        for j in range(1, Ap.shape[1] - 1):
            Ap[i][j] = -1 / Pe * ((p[i - 1][j] - 2 * p[i][j] + p[i + 1][j]) / h_x ** 2 + (
                    p[i][j - 1] - 2 * p[i][j] + p[i][j + 1]) / h_y ** 2)
            if v1_func(i, j) > 0:
                Ap[i][j] += v1_func(i, j) * (p[i][j] - p[i - 1][j]) / h_x
            else:
                Ap[i][j] += v1_func(i, j) * (p[i + 1][j] - p[i][j]) / h_x
            if v2_func(i, j) > 0:
                Ap[i][j] += v2_func(i, j) * (p[i][j] - p[i][j - 1]) / h_y
            else:
                Ap[i][j] += v2_func(i, j) * (p[i][j + 1] - p[i][j]) / h_y
    return Ap


# скалярное произведение вектор-матриц
def Scr(a, b):
    sum = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            sum += a[i][j] * b[i][j]
    return sum


# задание правой части
B2 = np.zeros((Nx + 1, Ny + 1))
for i in range(Ny + 1):
    B2[0][i] = g_func(0, i)
    B2[Nx][i] = g_func(Nx, i)
for i in range(Nx + 1):
    B2[i][0] = g_func(i, 0)
    B2[i][Ny] = g_func(i, Ny)
for i in range(1, Nx):
    for j in range(1, Ny):
        B2[i][j] = f_func(i, j)


def IOM_m(vec_b, m):
    solution = np.zeros((Nx + 1, Ny + 1))
    k = 1  # количество векторов, к которым будет ортогонален очередной вектор
    x0 = np.zeros((Nx + 1, Ny + 1))  # Начальное приближение
    # задание краевых значений
    for ik in range(Ny + 1):
        x0[0][ik] = g_func(0, ik)
        x0[Nx][ik] = g_func(Nx, ik)
    for jk in range(Nx + 1):
        x0[jk][0] = g_func(jk, 0)
        x0[jk][Ny] = g_func(jk, Ny)
    r0 = vec_b - multA(x0)  # вектор начальной невязки
    k = 0
    while abs(LA.norm(r0)) > 10 ** (-8):
        V = np.zeros((Nx + 1, (m + 1) * (Ny + 1)))  # матрица базисных векторов из пространства K
        H = np.zeros((m + 1, m))  # матрица коэффициентов ортогонализации
        r0 = vec_b - multA(x0)  # вектор начальной невязки
        beta = LA.norm(r0)  # норма начальной невязки
        V[:, :Ny + 1] = r0 / beta  # первый базисный вектор пространства K
        for j in range(1, m + 1):
            omega_j = multA(V[:, (j - 1) * (Ny + 1): j * (Ny + 1)])  # базисный вектор пространства L
            for i in range(max(1, j - k + 1), j + 1):
                H[i - 1][j - 1] = Scr(omega_j,
                                      V[:, (i - 1) * (Ny + 1): i * (Ny + 1)])  # вычисление коэффициента орт-ции
                omega_j = omega_j - H[i - 1][j - 1] * V[:, (i - 1) * (Ny + 1): i * (
                            Ny + 1)]  # орт-ция очередного базисного вектора про-ва L
            H[j][j - 1] = LA.norm(omega_j)  # норма орт-го вектора
            if abs(H[j][j - 1]) < 10 ** (-5):
                m = j
                break
            V[:, j * (Ny + 1): (j + 1) * (Ny + 1)] = omega_j / H[j][j - 1]  # вычисление следующего вектора про-ва K
        e_1 = np.zeros(m + 1)  # орт
        e_1[0] = 1
        g = beta * e_1  # вектор правой части вспопогательной СЛАУ
        H = np.c_[H, g]  # добавление к матрице системы правой части
        H = givens(H, m + 1)  # зануляем поддиагональ вращениями Гивенса
        g = H[:m, m]  # перезаписываем измененую правую часть
        H = np.delete(np.delete(H, m, 1), m, 0)  # удаляем вектор правой части из системы
        y = Gauss_back_step(H, g, m)  # обратный ход метода Гауса
        # Уточнение решения
        sumyivi = np.zeros((Nx + 1, Ny + 1))  # уточняющий вектор
        for f in range(1, m + 1):
            sumyivi += y[f - 1] * V[:, (f - 1) * (Ny + 1): f * (Ny + 1)]  # вычисление уточняющего вектора
        solution = x0 + sumyivi  # уточнение
        r0 = vec_b - multA(solution)  # вычисление вектора начальной невязки
        x0 = solution  # изменение начального приближения
        print(LA.norm(r0))
        k += 1
    return solution, r0, m, k


ts = time.time()
lol, nev, ver, iter = IOM_m(B2, 10)
fig, ax = plt.subplots(1, 1, figsize = (5, 5))
ax.pcolormesh(np.linspace(0, Nx * h_x, Nx + 2), np.linspace(0, Ny * h_y, Ny + 2), lol)
plt.show()
tf = time.time()
print(pd.DataFrame(lol).to_latex())
print('Итераций', iter)
print('Время = ', tf - ts)
print('Невязка = ', LA.norm(nev))
t = np.linspace(0, Nx * h_x, Nx + 1)
y = np.linspace(0, Ny * h_y, Ny + 1)
x_grid, y_grid = np.meshgrid(y, t)
fig = plt.figure(figsize=(15, 5))
ax = plt.subplot(131, projection='3d')

ax.set_title(r'$Au_h$')
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
ax.set_zlabel(r'z')
ax1 = plt.subplot(132, projection='3d')
ax2 = plt.subplot(133, projection='3d')
ax1.set_title(r'$b$')
ax1.set_xlabel(r'x')
ax1.set_ylabel(r'y')
ax1.set_zlabel(r'z')
surf = ax.plot_surface(x_grid, y_grid, multA(lol), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, label=r'$\|r\|$')
surf1 = ax1.plot_surface(x_grid, y_grid, B2, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
ax2.set_title(r'$u_h$')
ax2.set_xlabel(r'x')
ax2.set_ylabel(r'y')
ax2.set_zlabel(r'z')
surf2 = ax2.plot_surface(x_grid, y_grid, lol, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
fig.colorbar(surf2, shrink=0.5, location='left')
# plt.show()
