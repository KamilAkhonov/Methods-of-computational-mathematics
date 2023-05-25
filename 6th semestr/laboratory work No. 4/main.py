import numpy as np
import matplotlib.pyplot as plt
import time

sumiter = 0
numiter = 0

# Границы x и t
L = 1
T = 1
# Сетка
tau_grid = 0.0001
x_grid = 0.1
# сетка по x
Nx = int(L / x_grid + 1)
h = np.linspace(0, L, Nx)
# сетка по t
Nt = int(T / tau_grid + 1)
tau = np.linspace(0, T, Nt)


# Коэффициент теплопроводности
def Thermal_conductivity(t, x, y):
    return 1 + y ** (3 / 2)


# Тестовые примеры
# Плоскость
def tf1(t, x):
    return 2 * x + 3 * t + 5


def fx_tf1(t, x):
    return 3 - 6 * np.sqrt(2 * x + 3 * t + 5)


def yt0_tf1(t, x):
    return 2 * x + 5


def yx0_tf1(t, x):
    return 3 * t + 5


def yxL_tf1(t, x):
    return 2 * L + 3 * t + 5


# Линейно-квадратичная функция
def tf2(t, x):
    return 2 * x ** 2 + t + 3


def fx_tf2(t, x):
    return -3 - 4 * ((2 * x ** 2 + t + 3) ** (3 / 2)) - (24 * x ** 2) * np.sqrt(2 * x ** 2 + t + 3)


def yt0_tf2(t, x):
    return 2 * x ** 2 + 3


def yx0_tf2(t, x):
    return t + 3


def yxL_tf2(t, x):
    return 2 * L ** 2 + t + 3


# Параболоид
def tf3(t, x):
    return 3 * x ** 2 + 2 * t ** 2 + 1


def fx_tf3(t, x):
    return 4 * t - 6 - 6 * (tf3(t, x) ** 1.5) - 54 * (x ** 2) * (tf3(t, x) ** 0.5)


def yt0_tf3(t, x):
    return 3 * x ** 2 + 1


def yx0_tf3(t, x):
    return 2 * t ** 2 + 1


def yxL_tf3(t, x):
    return 3 * L ** 2 + 2 * t ** 2 + 1


# Индивидуальная функция
def tf4(t, x):
    return np.exp(np.sin(x) ** 2 + np.cos(t) ** 2)


def fx_tf4(t, x):
    return tf4(t, x) * (-1 * np.sin(2 * t) - 2 * np.cos(2 * x) * (1 + (tf4(t, x) ** 1.5)) - 1 * (np.sin(2 * x) ** 2)
                        * (1 + 2.5 * (tf4(t, x) ** 1.5)))


def yt0_tf4(t, x):
    return np.exp((np.sin(x) ** 2) + 1)


def yx0_tf4(t, x):
    return np.exp(np.cos(t) ** 2)


def yxL_tf4(t, x):
    return np.exp(np.sin(L) ** 2 + np.cos(t) ** 2)


# аппроксимация коэффициента теплопроводности
def a(t_1, x_1, y_1, t_2, x_2, y_2):
    return 0.5 * (Thermal_conductivity(t_1, x_1, y_1) + Thermal_conductivity(t_2, x_2, y_2))


# Чисто неявная схема
def pis(a, fp, yleft, yright, tm, phif, epsilon):
    # Искомый слой
    global sumiter
    global numiter
    res = np.zeros(Nx)
    # Начальное приближение
    for i in range(Nx):
        res[i] = fp[i]  # fp - предыдущий слой
    res[0] = yleft  # Краевое условие слева
    res[-1] = yright  # Краевое условие справа
    res_help = res.copy()  # вспомогательный слой
    res_const = res.copy()  # предыдущий слой
    res_prom = res.copy()  # слой после итерации
    k = 0  # счетчик
    while True:
        k += 1
        for i in range(1, Nx - 1):
            res_prom[i] = (tau_grid / x_grid ** 2) * (
                        a(tm, h[i + 1], res_help[i + 1], tm, h[i], res_help[i]) * (res_help[i + 1] - res_help[i]) - a(
                    tm, h[i], res_help[i], tm, h[i - 1], res_help[i - 1]) * (
                                    res_help[i] - res_help[i - 1])) + tau_grid * phif[i] + res_const[i]
        res_help = res_prom.copy()
        # проверка условия выхода из цикла итераций
        if np.linalg.norm(res_help - res) < epsilon:
            sumiter += k
            numiter += 1
            # print(k)  # вывод количества итераций
            return res_help
        else:
            res = res_help.copy()


# симметричная схема
def simsh(a, fp, yleft, yright, tm, phif, epsilon):
    # Вектор решения в определенный момент времени
    global sumiter
    global numiter
    res = np.zeros(Nx)
    # Начальное приближение
    for i in range(Nx):
        res[i] = fp[i]
    res_const = res.copy()  # предыдущий слой
    res[0] = yleft  # Краевое условие слева
    res[-1] = yright  # Краевое условие справа
    res_help = res.copy()  # вспомогательный слой
    res_prom = res.copy()  # слой после итерации
    k = 0  # счетчик
    while True:
        k += 1
        for i in range(1, Nx - 1):
            res_prom[i] = (tau_grid / (2 * x_grid ** 2)) * (
                        a(tm, h[i + 1], res_help[i + 1], tm, h[i], res_help[i]) * (res_help[i + 1] - res_help[i]) - a(
                    tm, h[i], res_help[i], tm, h[i - 1], res_help[i - 1]) * (res_help[i] - res_help[i - 1])) + (
                                      tau_grid / (2 * x_grid ** 2)) * (
                                      a(tm - tau_grid, h[i + 1], res_const[i + 1], tm - tau_grid, h[i],
                                        res_const[i]) * (res_const[i + 1] - res_const[i]) - a(tm - tau_grid, h[i],
                                                                                              res_const[i],
                                                                                              tm - tau_grid, h[i - 1],
                                                                                              res_const[i - 1]) * (
                                                  res_const[i] - res_const[i - 1])) + tau_grid * phif[i] + res_const[i]
        res_help = res_prom.copy()
        # проверка условия выхода из цикла итераций
        if np.linalg.norm(res_help - res) < epsilon:
            sumiter += k
            numiter += 1
            # print(k)  # вывод количества итераций
            return res_help
        else:
            res = res_help.copy()


def solve_pis(ut0, ux0, uxL, tf, ep):
    print('---Чисто неявная схема---')
    solve_matr = np.zeros((Nt, Nx))  # Матрица решений
    # заполнение матрицы начальным условием t = 0
    for i in range(Nx):
        solve_matr[Nt - 1][i] = ut0(0, h[i])
    fi = np.zeros(Nx)  # начальное приближение
    for tme in range(1, Nt):
        for j in range(Nx):  # заполнение начального приближения
            fi[j] = tf(tau[tme], h[j])
        s = pis(a, solve_matr[Nt - tme], ux0(tau[tme], h[0]), uxL(tau[tme], h[Nx - 1]), tau[tme], fi, ep)
        solve_matr[Nt - tme - 1] = s
    return solve_matr


def solve_simshame(ut0, ux0, uxL, tf, ep):
    print('---Симметричная схема---')
    solve_matr = np.zeros((Nt, Nx))  # Матрица решений
    # заполнение матрицы начальным условием t = 0
    for i in range(Nx):
        solve_matr[Nt - 1][i] = ut0(0, h[i])
    fi = np.zeros(Nx)  # начальное приближение
    for tme in range(1, Nt):
        for j in range(Nx):
            fi[j] = tf(tau[tme], h[j])  # заполнение начального приближения
        s = simsh(a, solve_matr[Nt - tme], ux0(tau[tme], h[0]), uxL(tau[tme], h[Nx - 1]), tau[tme], fi, ep)
        solve_matr[Nt - tme - 1] = s
    return solve_matr


# вычисление погрешности найденного решения
def error_pis(func, ut0, ux0, uxL, tf, ep):
    t_matr = np.zeros((Nt, Nx))
    for i in range(Nt - 1, -1, -1):
        for j in range(Nx):
            t_matr[i][j] = func(tau[Nt - i - 1], h[j])
    nt_matr = solve_pis(ut0, ux0, uxL, tf, ep)
    print(np.max(np.abs(nt_matr - t_matr)))


# вычисление погрешности найденного решения
def error_simsh(func, ut0, ux0, uxL, tf, ep):
    t_matr = np.zeros((Nt, Nx))
    for i in range(Nt - 1, -1, -1):
        for j in range(Nx):
            t_matr[i][j] = func(tau[Nt - i - 1], h[j])
    nt_matr = solve_simshame(ut0, ux0, uxL, tf, ep)
    print('Ошибка = ', np.max(np.abs(nt_matr - t_matr)))


# start_time = time.time()
# error_pis(tf4, yt0_tf4, yx0_tf4, yxL_tf4, fx_tf4, 0.000001)
# print('pis = ', time.time() - start_time)
# print('Число итераций в среднем = ', sumiter/numiter)
start_time = time.time()
error_simsh(tf4, yt0_tf4, yx0_tf4, yxL_tf4, fx_tf4, 0.000001)
print('sec = ', time.time() - start_time)
print('Число шагов в среднем = ', sumiter/numiter)

# t_matr = np.zeros((Nt, Nx))
# for i in range(Nt - 1, -1, -1):
#     for j in range(Nx):
#         t_matr[i][j] = tf4(tau[Nt - i - 1], h[j])
# # Compute z to make the pringle surface.
# xgrid, ygrid = np.meshgrid(h, tau)
# # z = solve_pis(yt0_tf3, yx0_tf3, yxL_tf3, fx_tf3, 0.1)
# zet = np.exp(np.sin(xgrid) ** 2 + np.cos(ygrid) ** 2)
# ax = plt.figure().add_subplot(projection='3d')
# print(t_matr)
# t_matr.swapaxes(1,1)
# print(t_matr)

# ax.plot_surface(xgrid, ygrid, t_matr, cmap='viridis')
# ax.plot_surface(xgrid, ygrid, zet, cmap='plasma')
#
# plt.show()
