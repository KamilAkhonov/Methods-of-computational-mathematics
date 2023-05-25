import numpy as np
import numpy.linalg as LinAl
import pandas as pd
import matplotlib as plt

num_list_group = 3  # номер в списке группы
N = 4  # размерность системы
K = N  # полуширина ленты
A = np.zeros((N, N))  # матрица A системы
elmin = max(N, 10)  # элемент главной диагонали
# формируем матрицу A
for i in range(N):
    for j in range(N):
        if i == j:
            A[i][j] = elmin
        if 0 <= i + j <= N - 1 and 1 <= j <= K:
            A[i][i + j] = 1 / j
        if 0 <= i - j <= N - 1 and 1 <= j <= K:
            A[i][i - j] = 1 / j
x = np.zeros(N)  # вектор решения
# формируем вектор решения
for i in range(N):
    if i == 0 or i == N - 1 or (i == num_list_group and 1 <= num_list_group < N - 1) or (
            i == (N - 1) - 2 * num_list_group and 1 <= (N - 1) - 2 * num_list_group < N - 1):
        x[i] = 1
B = np.zeros(N)  # вектор правой части
# формируем вектор правой части
B = np.dot(A, x)


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


def IOM_m(matr_A, vec_b, m):
    k = 1  # количество векторов, к которым будет ортогонален очередной вектор
    x0 = np.zeros(N)  # Начальное приближение
    r0 = vec_b - np.dot(matr_A, x0)  # вектор начальной невязки
    while abs(LinAl.norm(r0)) > 10 ** (-12):
        V = np.zeros((N, m + 1))  # матрица базисных векторов из пространства K
        H = np.zeros((m + 1, m))  # матрица коэффициентов ортогонализации
        r0 = vec_b - np.dot(matr_A, x0)
        beta = LinAl.norm(r0)  # норма начальной невязки
        V[:, 0] = r0 / beta  # первый базисный вектор пространства K
        for j in range(1, m + 1):
            omega_j = np.dot(A, V[:, j - 1])  # базисный вектор пространства L
            for i in range(max(1, j - k + 1), j + 1):
                H[i - 1][j - 1] = np.dot(np.transpose(omega_j), V[:, i - 1])  # вычисление коэффициента орт-ции
                omega_j = omega_j - H[i - 1][j - 1] * V[:, i - 1]  # орт-ция очередного базисного вектора про-ва L
            H[j][j - 1] = LinAl.norm(omega_j)  # норма орт-го вектора
            if abs(H[j][j - 1]) < 10 ** (-8):
                m = j
                break
            V[:, j] = omega_j / H[j][j - 1]  # вычисление следующего вектора про-ва K
        e_1 = np.zeros(m + 1)  # орт
        e_1[0] = 1
        g = beta * e_1  # вектор правой части вспопогательной СЛАУ
        H = np.c_[H, g]  # добавление к матрице системы правой части
        H = givens(H, m + 1)  # зануляем поддиагональ вращениями Гивенса
        g = H[:m, m]  # перезаписываем измененую правую часть
        H = np.delete(np.delete(H, m, 1), m, 0)  # удаляем вектор правой части из системы
        y = Gauss_back_step(H, g, m)  # обратный ход метода Гауса
        # Уточнение решения
        sumyivi = np.zeros(N)  # уточняющий вектор
        for i in range(m):
            sumyivi += y[i] * V[:, i]  # вычисление уточняющего вектора
        solution = x0 + sumyivi  # уточнение
        r0 = vec_b - np.dot(A, solution)    # вычисление вектора начальной невязки
        x0 = solution   # изменение начального приближения
    return solution, r0


sol, nv = IOM_m(A, B, 2)
print('Решение - ', sol)
print('Невязка - ', LinAl.norm(nv))

