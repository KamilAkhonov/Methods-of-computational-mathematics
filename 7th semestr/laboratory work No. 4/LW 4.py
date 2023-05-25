import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

# import pandas as pd


# ++++++++++++++++
h_x = 0.001
h_y = 0.001
Nx = 30
Ny = 20
Pe = 1
eps = 10 ** -5
# ****************

# ---------------------------------------------------------------------------------------------------- #
# Тест №1 - параболоид(ЛР3)                                                                            #
test = 1  #


# Тест №2 - квадратично-линейная функция(ЛР3)                                                          #
# test = 2                                                                                             #
# Тест №3 - Индивидульаная функция(ЛР3)                                                                #
# test = 3                                                                                             #
# Тест №4 - нелинейный пример <параболоид> (k2 = 1 + u, k1 = 1, v1 = 0, v2 = 0)                        #
# test = 4                                                                                             #
# Тест №5 - нелинейный пример <плоскость> (k2 = u^2 - 1, k1 = x - u, v1 = 1, v2 = 1)                   #
# test = 5                                                                                             #
# Тест №6 - нелинейный пример <произведение синусов> (k1 = sin(x) + u, k2 = 1, v1 = 0, v2 = 0)         #
# test = 6                                                                                             #
# ---------------------------------------------------------------------------------------------------- #
# test = 7

# -----------------------------------------------------------------------------------------------------#
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
        case 4:
            return 1 + ((x - 1) ** 2) + ((y - 2) ** 2)
        case 5:
            return x + y
        case 6:
            return np.sin(2 * x) * np.sin(y)
        case 7:
            return x + y


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
            # return 0
        case 4:
            return 0
        case 5:
            return 0
        case 6:
            return np.sin(2 * x) * np.sin(y)
        case 7:
            return 0


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
        case 4:
            # return -4 - 2 * np.sqrt(u_func(i, j)) - 1 * (2 * y ** 2 - 8 * y + 8) / np.sqrt(u_func(i, j))
            return -20 - u_func(i, j) - 2 * y ** 2 + 8 * y
        case 5:
            return 1
        case 6:
            return 4 * np.sin(y) ** 2 * (np.cos(4 * x) - np.sin(2 * x))
        case 6:
            return 2


def v1_func(i, j, u):
    x = i * h_x
    y = j * h_y
    match test:
        case 1:
            return 0
        case 2:
            return y - 1
        case 3:
            return 4 * y * np.sin(2 * y ** 2)
        case 4:
            return 0
        case 5:
            return 1

        case 6:
            return 0
        case 7:
            return 1


def v2_func(i, j, u):
    x = i * h_x
    y = j * h_y
    match test:
        case 1:
            return 0
        case 2:
            return 2 * x
        case 3:
            return x * np.cos(x ** 2)
        case 4:
            return 0
        case 5:
            return 1
        case 6:
            return 0
        case 7:
            return 1


def k1_func(i, j, u):
    x = i * h_x
    y = j * h_y
    match test:
        case 1:
            return 1
        case 2:
            return 1
        case 3:
            return 1
        case 4:
            return 1
        case 5:
            return x - u
        case 6:
            return np.sin(y) + u
        case 7:
            return 1


def k2_func(i, j, u):
    match test:
        case 1:
            return 1
        case 2:
            return 1
        case 3:
            return 1
        case 4:
            return 1 + u
        case 5:
            return u ** 2 - 1
        case 6:
            return 1
        case 7:
            return 1


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


def Func(arg):
    result = np.zeros(arg.shape)
    for i in range(1, arg.shape[0] - 1):
        for j in range(1, arg.shape[1] - 1):
            if np.abs(0.5 * (arg[i][j] + arg[i - 1][j])) < 10 ** (-8):
                valU = 0
            else:
                valU = 0.5 * (arg[i][j] + arg[i - 1][j])
            if np.abs(0.5 * (arg[i][j] + arg[i][j - 1])) < 10 ** (-8):
                valU2 = 0
            else:
                valU2 = 0.5 * (arg[i][j] + arg[i][j - 1])
            result[i][j] -= (k1_func(i + 0.5, j, 0.5 * (arg[i + 1][j] + arg[i][j])) * (arg[i + 1][j] - arg[i][j]) -
                             k2_func(i - 0.5, j, valU) * (
                                     arg[i][j] - arg[i - 1][j])) / h_x ** 2
            result[i][j] -= (k1_func(i, j + 0.5, 0.5 * (arg[i][j + 1] + arg[i][j])) * (arg[i][j + 1] - arg[i][j])
                             - k2_func(i, j - 0.5, valU2) * (
                                     arg[i][j] - arg[i][j - 1])) / h_y ** 2
            if v1_func(i, j, arg[i][j]) > 0:
                result[i][j] += v1_func(i, j, arg[i][j]) * (arg[i][j] - arg[i - 1][j]) / h_x
            else:
                result[i][j] += v1_func(i, j, arg[i][j]) * (arg[i + 1][j] - arg[i][j]) / h_x
            if v2_func(i, j, arg[i][j]) > 0:
                result[i][j] += v2_func(i, j, arg[i][j]) * (arg[i][j] - arg[i][j - 1]) / h_y
            else:
                result[i][j] += v2_func(i, j, arg[i][j]) * (arg[i][j + 1] - arg[i][j]) / h_y
    return result - B2


# задание правой части


def DhF(x, w):
    machine_epsilon = 10 ** -17
    h_opt = np.sqrt(machine_epsilon)
    if np.abs(np.linalg.norm(w)) > 10 ** (-8):
        if np.abs(np.linalg.norm(x)) > 10 ** (-8):
            return (Func(x + h_opt * w * np.linalg.norm(x) / np.linalg.norm(w)) - Func(x)) / \
                   (h_opt * np.linalg.norm(x) / np.linalg.norm(w))
        else:
            return (Func(h_opt * w / np.linalg.norm(w)) - Func(np.zeros(x.shape))) / (h_opt * np.linalg.norm(w))
    else:
        return np.zeros(x.shape)


def IOM_m(vec_b, m):
    solution = np.zeros((Nx + 1, Ny + 1))
    k = 1  # количество векторов, к которым будет ортогонален очередной вектор
    # задание краевых значений
    V = np.zeros((Nx + 1, (m + 1) * (Ny + 1)))  # матрица базисных векторов из пространства K
    H = np.zeros((m + 1, m))  # матрица коэффициентов ортогонализации
    r0 = vec_b  # вектор начальной невязки
    beta = LA.norm(r0)  # норма начальной невязки
    V[:, :Ny + 1] = r0 / beta  # первый базисный вектор пространства K
    for j in range(1, m + 1):
        omega_j = DhF(solution, V[:, (j - 1) * (Ny + 1): j * (Ny + 1)])  # базисный вектор пространства L
        for i in range(max(1, j - k + 1), j + 1):
            H[i - 1][j - 1] = Scr(omega_j, V[:, (i - 1) * (Ny + 1): i * (Ny + 1)])  # вычисление коэффициента орт-ции
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
    solution += sumyivi  # уточнение
    return solution


def JacFreeNK():
    x0 = np.full((Nx + 1, Ny + 1), 0)  # Начальное приближение
    # задание краевых значений
    for i in range(Ny + 1):
        x0[0][i] = g_func(0, i)
        x0[Nx][i] = g_func(Nx, i)
    for i in range(Nx + 1):
        x0[i][0] = g_func(i, 0)
        x0[i][Ny] = g_func(i, Ny)
    delta = 1
    while np.linalg.norm(delta) > eps:
        b = -1 * Func(x0)
        delta = IOM_m(b, 100)
        x0 = x0 + delta
        print('Delta = ', np.linalg.norm(delta))
    return x0


# toch = solve_of_task(u_func)

resh = JacFreeNK()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.pcolormesh(np.linspace(0, Ny * h_y, Ny + 1), np.linspace(0, Nx * h_x, Nx + 1), resh)
plt.show()
r1 = -Func(resh)
print("Невязка = ", LA.norm(r1))
# print(pd.DataFrame(resh).to_latex())
