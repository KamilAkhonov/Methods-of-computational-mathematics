import numpy as np
import pandas as pd
import time

# Число столбцов в матрице A
N = 1000
# Вектор перестановок
trans = np.linspace(1, N, N)
#  Ввод матрицы A
A = np.zeros((N + 1, N))
for i in range(N + 1):
    for j in range(N):
        A[i][j] = i + j - 2 * i * j
#  ММ.ГГГГ
happy_birthday = [0, 1, 1, 8, 3, 8]
for i in range(6):
    A[i][N - 1] = happy_birthday[i]
# копия матрицы A
Atr = A.copy()
# Тестовый пример
# Вектор точного решения
solution = np.zeros(N)
for i in range(N):
    solution[i] = 1
# Вектор правой части
b = np.dot(A, solution)
# Представление в виде столбца N+1 x 1
b = b.reshape((N + 1, 1))
# Копирование вектора правой части
btr = np.copy(b)
btr = btr.reshape(N + 1)

# Инициализация матрицы Q
Q = np.eye(N + 1)
# Матрица перестановок
Matr_of_swap = np.eye(N)


# функция перестановки столбцов в матрице
def swap(A, n1, n2):
    if n1 != n2:
        # Перестановка в матрице A
        temp1 = A[:, n1].copy()
        temp2 = A[:, n2].copy()
        A[:, n2] = temp1[:]
        A[:, n1] = temp2[:]
        # Перестановка в матрице перестановок
        temp3 = trans[n1 - 1]
        trans[n1 - 1] = trans[n2 - 1]
        trans[n2 - 1] = temp3


# Отражения Хаусхолдера
def householder_mod(A, b):
    # допуск к измененнию локальной переменной, как к глобальной
    global Q
    for k in range(N):
        print(k + 1, 'Шаг Хаусхолдера')
        # номер ведущего столбца
        h = -1
        # норма ведущего столбца
        sr = -1
        # df = pd.DataFrame(A)
        # print(df.to_latex())
        # pd.DataFrame(A).to_latex(buf = "file.tex", index_names = False)
        # minor = A[k:N + 1, k:N + 1]
        # np.savetxt("foo.csv", minor, delimiter=",")
        # выбор ведущего столбца
        for j in range(k, N):
            print("Норма", j - k, "столбца = ", format(np.linalg.norm(A[k:, j]), '.7'), "\\\ ")
            # print(A[k:, j])
            if np.linalg.norm(A[k:, j]) > sr:
                h = j
                sr = np.linalg.norm(A[k:, j])
        # Условие на проверку равенства нулю всех столбцов на k-шаге алгоритма
        if sr <= 10 ** (-8):
            break
        # Перестановка ведущего столбца на первое место на k-шаге в матрице A
        swap(A, k, h)
        # Перестановка столбцов в матрице перестановок
        # swap(Matr_of_swap, k, h)
        # ведущий столбец
        main_column = A[k:, k]
        # норма ведущего столбца
        norm_main_column = sr
        # единичный базисный вектор
        e_1 = np.zeros(len(main_column))
        e_1[0] = 1
        # вектор w в отражениях Хаусхолдера
        w = np.zeros((1, N + 1 - k))
        # вычисление w, исходя из условия на выбор знака
        if main_column[0] >= 0:
            w[0] = main_column + norm_main_column * e_1
        else:
            w[0] = main_column - norm_main_column * e_1

        # Матрица Хаусхолдера
        # mainE = np.eye(N + 1)
        # mainE[k:N + 1, k:N + 1] = np.eye(N + 1 - k) - 2 / (np.linalg.norm(w[0]) ** 2) * np.dot(np.transpose(w), w)
        # вычисление матрицы Q
        # Q = np.dot(Q, mainE)

        # A^T * w
        b_help = np.dot(np.transpose(A[k:N + 1, k:N + 1]), np.transpose(w))
        # (w, b)
        bb_help = np.dot(w, b[k:N + 1])
        # 2/(w,w)
        beta_help = 2 / np.dot(w, np.transpose(w))[0][0]
        # Поэлементно вычитаем из вектора правой части и матрицы
        # С условием проверки малости значений и присваиванием им строго нулевого значения
        for i in range(k, N + 1):
            b[i] -= beta_help * np.dot(bb_help, w)[0][i - k]
            # if abs(b[i] - beta_help * np.dot(bb_help, w)[0][i - k]) < 10 ** (-8):
            #     b[i] = 0
            # else:
            #     b[i] -= beta_help * np.dot(bb_help, w)[0][i - k]
            for j in range(k, N):
                A[i][j] -= beta_help * np.dot(np.transpose(w), np.transpose(b_help))[i - k][j - k]
                # if abs(A[i][j] - beta_help * np.dot(np.transpose(w), np.transpose(b_help))[i - k][j - k]) < 10 ** (-8):
                #     A[i][j] = 0
                # else:
                #     A[i][j] -= beta_help * np.dot(np.transpose(w), np.transpose(b_help))[i - k][j - k]
        # print(k + 1, 'Итерация')
        # print('Очередная матрица с зануленным столбцом')
        # print(df.to_latex())
        # print(b)
    # Q[k:N+1,k:N+1] = np.eye(N+1 - k)
    return A, b

# обратный ход метода Гаусса
def gauss(A, B):
    sol = np.zeros(len(A[0]))
    for i in range(len(A[0]) - 1, -1, -1):
        s = 0
        if i == len(A[0]) - 1:
            sol[i] = B[i] / A[i][i]
        else:
            for j in range(i + 1, len(A[0]), 1):
                s += A[i][j] * sol[j]
            sol[i] = (B[i] - s) / A[i][i]
    return sol

# вычисление ранга треугльной матрицы
def rank_matrix(A):
    rank = 0
    for i in range(len(A)):
        if abs(A[i][i]) <= 10 ** (-8) :
            break
        else:
            rank += 1
    return rank

# функция перестановки компонент вектора решения
def trsol(pst, sol):
    true_sol = np.zeros(len(pst))
    for i in range(len(pst)):
        true_sol[pst[i] - 1] = sol[i]
    return true_sol

# Решение методом SVD разложения
def svd_solve(A, B):
    # Инициализация вектора решения
    solution = np.zeros(len(A[0]))
    # SVD разложение
    U, S, V_t = np.linalg.svd(Atr)
    # ранг матрицы
    r = 0
    # вспомогательный вектор y
    y = np.zeros(len(A[0]))
    # Вычисление ранга по сингулярным числам
    for i in range(len(S)):
        if S[i] > 10 ** (-8):
            r += 1
    # Вспомогательный вектор с = U^T * b
    c = np.dot(np.transpose(U), B)
    # вычисление вектора y
    for i in range(r):
        y[i] = c[i] / S[i]
    # дополнение нулями компонент с r + 1 до n
    y.resize(N)
    # Псевдообратная матрица к матрице \Sigma^+
    sigm = np.zeros((N + 1, N))
    for i in range(r):
        sigm[i][i] = S[i]
    # вычисление решения
    solution = np.dot(np.transpose(V_t), y)
    # вычисление точности
    acurrancy = (np.linalg.norm(np.dot(sigm, y).reshape((N + 1, 1)) - c)) ** 2
    return solution, acurrancy

# Получение псевдообратной матрицы
def Generalized_inverse(A):
    # SVD разложение
    U, S, V_t = np.linalg.svd(A)
    # вычисление псевдообратной матрицы \Sigma^+
    sigm_plus = np.zeros((N, N + 1))
    for i in range(3):
        sigm_plus[i][i] = 1 / S[i]
    return np.dot(np.dot(np.transpose(V_t), sigm_plus), np.transpose(U))


assvaadaa = time.time()
R, btild = householder_mod(A, b)
rank_R = rank_matrix(R)
answ = gauss(R[:rank_R, :rank_R], btild[:rank_R])
answ.resize(N)
print('Время - ', time.time() - assvaadaa)
btild = np.reshape(btild, N + 1)
print(answ)
print('Точность - ', np.linalg.norm(np.dot(R, answ) - btild) ** 2)

# assvaadaa = time.time()
# sol, acc = svd_solve(A,b)
# print('Решение = ', sol)
# print('Точность = ', acc)
# print('Время = ', time.time() - assvaadaa)

# R, btild = householder_mod(A, b)
# print('1111111111111111111111111111111111111111111111111111111111111111111111')
# U, S, V = np.linalg.svd(Atr)
# df = pd.DataFrame(np.dot(np.transpose(U),U))
# print(df.to_latex())
# df = pd.DataFrame(np.dot(U,np.transpose(U)))
# print(df.to_latex())
# sigm = np.zeros((N+1, N))
# for i in range(3):
#     sigm[i][i] = S[i]
# print('+--+-+-+-+-+-+-+-')
# print(pd.DataFrame(Generalized_inverse(Atr)).to_latex())
# print(pd.DataFrame(np.dot(A, Generalized_inverse(A))).to_latex())
# print(pd.DataFrame(np.dot(Generalized_inverse(A), A)).to_latex())


#
# print(pd.DataFrame(sigm).to_latex())
# print(pd.DataFrame(np.dot(np.transpose(V), V)).to_latex())
# print(pd.DataFrame(np.dot(V, np.transpose(V))).to_latex())
# print('Solve by SVD', pd.DataFrame(svd_solve(A,b)).to_latex())
#
# print('Check equals', pd.DataFrame(np.dot((np.dot(np.transpose(U),Atr)),np.transpose(V)) - sigm).to_latex())
# print(np.dot(R, solution))
# print(btild)
# print('Матрица Q', Q)
# rank_R = rank_matrix(R)
# answ = gauss(R[:rank_R, :rank_R], btild[:rank_R])
# answ.resize(N)
# print(trans)
# print(solution)
print('------------------------------------')
# btild = np.reshape(btild,N+1)
# # print(answ)
# print(np.linalg.norm(np.dot(R,answ) - btild))
# print(trsol(trans,answ))
# print(Atr)
# print(trans)
# print(btr)
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
# df = pd.DataFrame(np.dot(Atr,Matr_of_swap) - np.dot(Q,R))
# print(df.to_latex())
# print(Atr)
# gh = pd.DataFrame(Matr_of_swap)
#
# print(gh.to_latex())
# # np.dot(np.transpose(Q), R))
# print(np.dot(Atr,trsol(trans,answ)) - btr)
print('----------------------------------------------------------------')
