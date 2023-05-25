import numpy as np

# начальные данные
L_1 = 2
L_2 = 3
T = 5 * (L_1 + L_2)
N_x = 161  # 21 41 81 161
N_y = 241  # 31 61 121 241
N_t = 101  # 26 51 101 201
a = 1


# метод прогонки
def TDMA(a, b, c, f):
    a, b, c, f = tuple(
        map(lambda k_list: list(map(float, k_list)),
            (a, b, c, f)))

    alpha = [-b[0] / c[0]]
    beta = [f[0] / c[0]]
    n = len(f)
    x = [0] * n

    for i in range(1, n):
        alpha.append(-b[i] / (a[i] * alpha[i - 1] + c[i]))
        beta.append((f[i] - a[i] * beta[i - 1]) / (
                a[i] * alpha[i - 1] + c[i]))

    x[n - 1] = beta[n - 1]

    for i in range(n - 1, 0, -1):
        x[i - 1] = alpha[i - 1] * x[i] + beta[i - 1]

    return x


# Тестовый пример №1
def f(x, y, t):
    return 5


def es(x, y, t):
    return 2 * x + 3 * y + 5 * t


def phi(x, y, t=0):
    return 2 * x + 3 * y + t


# тестовый пример 2

def f_2(x, y, t):
    return -4 * a - 3


def es_2(x, y, t):
    return 2 * x ** 2 + y - 3 * t + 2


def phi_2(x, y, t=0):
    return 2 * x ** 2 + y - 3 * t + 2


# тестовый пример 3
def f_3(x, y, t):
    return 1 - 2 * a


def es_3(x, y, t):
    return 3 * x ** 2 - 2 * y ** 2 + t + 4


def phi_3(x, y, t=0):
    return 3 * x ** 2 - 2 * y ** 2 + t + 4


# тестовый пример 4

# def f_4(x, y, t):
#     return 1 + a * (np.exp((np.sin(x)) ** 2
#                            + (np.cos(y)) ** 2)) * ((np.sin(2 * x)) ** 2 + 2 * np.cos(2 * x) -
#                                                    2 * np.cos(2 * y) + (np.sin(2 * x)) ** 2)
#
#
# def es_4(x, y, t):
#     return (np.exp((np.sin(x)) ** 2 + (np.cos(y)) ** 2)) + t
#
#
# def phi_4(x, y, t=0):
#     return (np.exp((np.sin(x)) ** 2 + (np.cos(y)) ** 2)) + t
def f_5(x, y, t):
    return 1 - a * (24 * x - 6 * y)


def es_5(x, y, t):
    return 4 * x ** 3 - y ** 3 + t - 2


def phi_5(x, y, t=0):
    return 4 * x ** 3 - y ** 3 + t - 2


def LODS_2(function_f, function_g, function_phi):
    # шаг по сетке <<x>>
    h_x = L_1 / (N_x - 1)
    # шаг по сетке <<y>>
    h_y = L_2 / (N_y - 1)
    # шаг по сетке <<t>>
    tau = T / (N_t - 1)

    # полученное решение
    solution = np.zeros((N_t, N_x, N_y))

    # Определим сетки
    x_grid = np.linspace(0, L_1, N_x)
    y_grid = np.linspace(0, L_2, N_y)
    tau_grid = np.linspace(0, T, N_t)

    # заполним <<нулевой слой>>
    for i in range(0, N_x):
        for j in range(0, N_y):
            solution[0][i][j] = function_phi(x_grid[i],
                                             y_grid[j])

    def UTilda(num_k):
        # Решим систему для первого этапа
        # главная диагональ
        md = np.full(N_x - 2, -1 * (h_x ** 2) - 2 * a * tau)
        # наддиагональ
        ud = np.full(N_x - 2, a * tau)
        # поддиагональ
        dd = np.full(N_x - 2, a * tau)
        dd[0] = 0
        ud[N_x - 3] = 0

        # Вспомогательный слой
        U_tilda = np.zeros((N_x, N_y))

        # Краевые точки
        U_tilda_0 = 0
        U_tilda_N = 0

        for k in range(1, N_y - 1):
            # вектор правой части
            b = np.zeros(N_x - 2)
            # заполним вектор правой части
            for l in range(0, N_x - 2):
                if l == 0:
                    b[l] = -1 * tau * (h_x ** 2) * function_f(x_grid[l + 1], y_grid[k], tau_grid[num_k + 1]) - 1 * (
                            h_x ** 2) * solution[num_k][l + 1][k] + ((a * tau / h_y) ** 2) * (
                                   function_g(x_grid[l], y_grid[k - 1], tau_grid[num_k + 1]) -
                                   (2 + (h_y ** 2 / (a * tau))) * function_g(x_grid[l], y_grid[k],
                                                                             tau_grid[num_k + 1]) +
                                   function_g(x_grid[l], y_grid[k + 1], tau_grid[num_k + 1]))
                    U_tilda_0 = (-1 * a * tau / (h_y ** 2)) * (
                            function_g(x_grid[l], y_grid[k - 1], tau_grid[num_k + 1]) -
                            (2 + (h_y ** 2 / (a * tau))) * function_g(x_grid[l], y_grid[k], tau_grid[num_k + 1]) +
                            function_g(x_grid[l], y_grid[k + 1], tau_grid[num_k + 1]))
                elif l == N_x - 3:
                    b[l] = -1 * tau * (h_x ** 2) * function_f(x_grid[l], y_grid[k], tau_grid[num_k + 1]) - (h_x ** 2) * \
                           solution[num_k][l][k] + ((a * tau / h_y) ** 2) * (
                                   function_g(x_grid[l + 1], y_grid[k - 1], tau_grid[num_k + 1]) -
                                   (2 + (h_y ** 2 / (a * tau))) * function_g(x_grid[l + 1], y_grid[k],
                                                                             tau_grid[num_k + 1]) +
                                   function_g(x_grid[l + 1], y_grid[k + 1], tau_grid[num_k + 1]))
                    U_tilda_N = (-1 * a * tau / (h_y ** 2)) * (
                            function_g(x_grid[l + 1], y_grid[k - 1], tau_grid[num_k + 1]) -
                            (2 + (h_y ** 2 / (a * tau))) * function_g(x_grid[l + 1], y_grid[k], tau_grid[num_k + 1]) +
                            function_g(x_grid[l + 1], y_grid[k + 1], tau_grid[num_k + 1]))
                else:
                    b[l] = -1 * tau * (h_x ** 2) * function_f(x_grid[l + 1], y_grid[k], tau_grid[num_k + 1]) - (
                            h_x ** 2) * \
                           solution[num_k][l + 1][k]
            # решим методом прогонки СЛАУ
            res = TDMA(dd, ud, md, b)
            # заносим решение во вспомогательный слой
            U_tilda[0][k] = U_tilda_0
            U_tilda[N_x - 1][k] = U_tilda_N
            for i in range(1, N_x - 1):
                U_tilda[i][k] = res[i - 1]
        return U_tilda

    def gtes(num_k):
        # Решим систему для второго этапа
        # главная диагональ
        md = np.full(N_y - 2, -1 * (2 * a * tau / (h_y ** 2) + 1))
        # наддиагональ
        ud = np.full(N_y - 2, a * tau / (h_y ** 2))
        # поддиагональ
        dd = np.full(N_y - 2, a * tau / (h_y ** 2))
        dd[0] = 0
        ud[N_y - 3] = 0

        # приближенное решение
        U = np.zeros((N_x, N_y))
        U_0 = 0
        U_N = 0

        # получим вспомогательный слой
        U_support = UTilda(num_k - 1)

        for k in range(0, N_x):
            # вектор правой части
            b = np.zeros(N_y - 2)
            # заполним вектор правой части
            for l in range(0, N_y - 2):
                if l == 0:
                    b[l] = -1 * U_support[k][l + 1] - (a * tau / (h_y ** 2)) * function_g(x_grid[k], y_grid[l],
                                                                                          tau_grid[num_k])
                    U_0 = function_g(x_grid[k], y_grid[l], tau_grid[num_k])
                elif l == N_y - 3:
                    b[l] = -1 * U_support[k][l] - (a * tau / (h_y ** 2)) * function_g(x_grid[k], y_grid[l + 1],
                                                                                      tau_grid[num_k])
                    U_N = function_g(x_grid[k], y_grid[l + 1], tau_grid[num_k])
                else:
                    b[l] = -1 * U_support[k][l + 1]
            # решим методом прогонки СЛАУ
            res = TDMA(dd, ud, md, b)
            # заносим решение
            U[k][0] = U_0
            U[k][N_y - 1] = U_N
            for i in range(1, N_y - 1):
                U[k][i] = res[i - 1]
            if k == 0:
                for j in range(1, N_y):
                    U[k][j] = function_g(x_grid[k], y_grid[j], tau_grid[num_k])
            if k == N_x-1:
                for j in range(1, N_y):
                    U[k][j] = function_g(x_grid[k], y_grid[j], tau_grid[num_k])
        return U

    for d in range(1, N_t):
        solution[d] = gtes(d)
    return solution


# получение тензора точного решения
def the_exact_solution(UFunc):
    solution = np.zeros((N_t, N_x, N_y))
    x_grid = np.linspace(0, L_1, N_x)
    y_grid = np.linspace(0, L_2, N_y)
    tau_grid = np.linspace(0, T, N_t)
    for depth in range(0, N_t):
        for x_axis in range(0, N_x):
            for y_axis in range(0, N_y):
                solution[depth][x_axis][y_axis] = UFunc(x_grid[x_axis], y_grid[y_axis], tau_grid[depth])
    return solution


# вычисление погрешности
def error(Ux, Ua):
    return np.max(np.abs(Ux - Ua))


exact_result = the_exact_solution(es_5)
# вычисление приближенного решения
result = LODS_2(f_5, es_5, phi_5)
# вычисление точного решения


print("Шаг по t -- ", T / (N_t - 1))
print("Шаг по x -- ", L_1 / (N_x - 1))
print("Шаг по y -- ", L_2 / (N_y - 1))
print("Ошибка -- ", error(result, exact_result))
