import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit

def Givens_qr(N):
    num_list_group = 3  # номер в списке группы
    # N = 1000 # размерность системы
    K = N   # полуширина ленты
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
    A = np.c_[A, B]  # дополняем матрицу системы вектором правой части
    # вращения Гивенса
    for l in range(N - 1):
        for i in range(N - 1, 0 + l, -1):
            j = i - 1
            if A[i][l] != 0:
                # вычислим коэффициенты C и S
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
                # Произведем вращение =)
                A[i], A[j] = A[i] * C - A[j] * S, A[j] * C + A[i] * S
                # A[j] = A[j] * C + A[i] * S
                # A[i] = temp
    B = A[:, N]  # извлекаем вектор правой части из матрицы A
    A = np.delete(A, N, 1)  # удаляем из матрицы A добавленный столбец
    # вектор решения
    sol = np.zeros(N)
    for i in range(N - 1, -1, -1):
        s = 0
        if i == N - 1:
            sol[i] = B[i] / A[i][i]
        else:
            for j in range(i + 1, N, 1):
                s += A[i][j] * sol[j]
            sol[i] = (B[i] - s) / A[i][i]
    return sol, x


# time_true = np.zeros(40)
# total = 0
# for i in range(len(dimension)):
#     tme, resh, toch = Givens_qr(int(dimension[i]))
#     print(int(dimension[i]), '&', tme, '&', np.linalg.norm(resh - toch), ' \ \ \hline')
#     print(resh)
resh, toch = Givens_qr(100)
print(np.linalg.norm(resh - toch))
# plt.grid()  # show grid
# plt.plot(dimension, time_iter, label='T(N)')
# plt.plot(dimension, time_true, label='N^3')
# plt.legend()
# plt.show()
