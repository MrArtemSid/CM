import numpy as np
from math import log, ceil


def print_row(row, p = 4):
    print(" ".join(f"{elem:8.{p}f}" for elem in row))


def print_matrix(matrix, p = 4):
    for row in matrix:
        print_row(row, p)


def to_triangle(matrix, lvl):
    if lvl >= len(matrix):
        return matrix

    for i in range(lvl, len(matrix)):
        vectorise = matrix[i][lvl - 1] / matrix[lvl - 1][lvl - 1]
        for j in range(lvl - 1, len(matrix[i])):
            matrix[i][j] -= matrix[lvl - 1][j] * vectorise
            if abs(matrix[i][j]) <= 1e-9:
                matrix[i][j] = 0

    return to_triangle(matrix, lvl + 1)


def find_solutions(matrix):
    n = len(matrix)
    solutions = [0] * n

    for k in range(n - 1, -1, -1):
        b_k = matrix[k][-1]
        sum_ax = sum(matrix[k][j] * solutions[j] for j in range(k + 1, n))
        solutions[k] = (b_k - sum_ax) / matrix[k][k]

    return solutions


def to_inverse_matrix(matrix):
    n = len(matrix)
    inverse = []

    for i in range(n):
        tmp_matrix = [row[:] for row in matrix]

        for j in range(n):
            tmp_matrix[j].append(j == i)

        triangle_matrix = to_triangle(tmp_matrix, 1)
        solutions = find_solutions(triangle_matrix)

        inverse.append(solutions)

    return np.array(inverse).T


def matrix_norm(matrix, size, p = 4):
    abs_matrix = np.abs(matrix).tolist()
    sums_of_rows = tuple(round(sum(row[:size]), p) for row in abs_matrix)
    ans = max(sums_of_rows)
    return f"max{sums_of_rows} = {ans:.{p}f}", ans


def vector_norm(matrix, p = 4):
    max_el = abs(matrix[0][-1])
    for i in range(1, len(matrix)):
        max_el = max(max_el, matrix[i][-1])
    return f"max{tuple(round(abs(el[-1]), p) for el in matrix)} = {max_el:.{p}f}", max_el


def gauss_method(matrix, delta):
    triangle_matrix = [row[:] for row in matrix]
    triangle_matrix = to_triangle(triangle_matrix, 1)

    print("Матрица в треугольном виде")
    print_matrix(triangle_matrix)

    solutions = find_solutions(triangle_matrix)
    print("Решения матрицы")
    print_row(solutions)

    print("Обратная матрица")
    inverse_matrix = to_inverse_matrix(matrix)
    print_matrix(inverse_matrix)

    print()
    for_print, norm_matrix = matrix_norm(matrix, len(matrix[0]) - 1)
    print(f"||A|| = {for_print}")
    for_print, norm_inv_matrix = matrix_norm(inverse_matrix, len(inverse_matrix[0]))
    print(f"||A^(-1)|| = {for_print}")

    for_print, norm_b = vector_norm(matrix)
    print(f"||b|| = {for_print}")

    float_p = ceil(abs(log(delta, 10)))

    rel_err_b = delta / norm_b
    print(
        f"Относительная погрешность вектора b (∆b / ||b||) = {delta:.{float_p}f} / {norm_b:.{float_p}f} <= {rel_err_b:.{float_p}f}")

    rel_err_sol = norm_inv_matrix * delta
    print(
        f"Относительная погрешность решения (||A^(-1)|| * ∆b) = {norm_inv_matrix:.{float_p}f} * {delta:.{float_p}f} <= {rel_err_sol:.{float_p + 1}f}")

    abs_err = norm_matrix * norm_inv_matrix * rel_err_b
    print(
        f"Абсолютная погрешность решения (||A|| * ||A^(-1)|| * δb) = {norm_matrix:.{float_p}f} * {norm_inv_matrix:.{float_p}f} * {rel_err_b:.{float_p}f} <= {abs_err:.{float_p + 1}f}")


def jacobi_convertion(matrix):
    converted_matrix = [row[:] for row in matrix]
    for i in range(len(matrix)):
        converted_matrix[i] = [-el / converted_matrix[i][i] for el in converted_matrix[i]]
        converted_matrix[i][i] = 0
    return converted_matrix


def check_jacobi(matrix, p):
    ret = False
    for i in range(len(matrix)):
        sum_el = sum(matrix[i]) - matrix[i][i]
        print(f"{matrix[i][i]} >= {" + ".join([f"{matrix[i][j]:.{p}f}" for j in range(len(matrix)) if j != i])}")
        if sum_el >= matrix[i][i]:
            ret = True
            break
    else:
        print("Соответствует правилу сходимости")
    return ret


def do_iterations(jacobi_matrix, cnt, delta):
    iterations = [[0] * (len(jacobi_matrix[0]) - 1), [0] * (len(jacobi_matrix[0]) - 1)]
    round_cnt = ceil(abs(log(delta, 10)))
    required_cnt = -1
    diff_ans = 0
    ans = []

    for k in range(cnt):
        for i in range(len(jacobi_matrix)):
            x_new = jacobi_matrix[i][-1]
            for j in range(len(jacobi_matrix[i]) - 1):
                if i != j:
                    x_new += jacobi_matrix[i][j] * iterations[0][j]
            iterations[1][i] = x_new

        max_diff = 0
        for z in range(len(iterations[0])):
            max_diff = max(max_diff, abs(iterations[1][z] - iterations[0][z]))

        if round(max_diff, round_cnt + 1) <= round(delta, round_cnt + 1) and required_cnt == -1:
            required_cnt = k
            ans = iterations[1].copy()
            diff_ans = max_diff

        iterations[0] = iterations[1].copy()

        print(k + 1, *np.round(iterations[-1], round_cnt + 1), round(max_diff, round_cnt + 1))
    print()
    print(f"Для получения нужной точности {delta} потребовалось {required_cnt + 1} итерации")
    return ans, diff_ans


def do_iterations_seidel(jacobi_matrix, delta):
    iterations = [[0] * (len(jacobi_matrix[0]) - 1), [0] * (len(jacobi_matrix[0]) - 1)]
    round_cnt = ceil(abs(log(delta, 10)))
    max_diff = 10
    cnt = 0

    while max_diff > delta:
        iterations[0] = np.copy(iterations[1])
        for i in range(len(jacobi_matrix)):
            iterations[1][i] = jacobi_matrix[i][-1] + sum(
                [jacobi_matrix[i][j] * iterations[1][j] for j in range(len(jacobi_matrix[i]) - 1) if i != j])
        cnt += 1
        max_diff = max(abs(iterations[0] - iterations[1]))
        print(cnt, *np.round(iterations[-1], round_cnt), f"{max_diff:.{round_cnt + 1}f}")

    print()
    print(f"Для получения нужной точности {delta} потребовалось {cnt} операций")

    return iterations[-1], round(max_diff, round_cnt + 1)


def jacobi_method(matrix, delta, is_seidel=0):
    round_cnt = ceil(abs(log(delta, 10)))
    print("Матрица для метода Якоби")
    new_matrix = jacobi_convertion(matrix)
    print_matrix(new_matrix, round_cnt)

    for_print, norm_B = matrix_norm(new_matrix, len(new_matrix[0]) - 1, round_cnt)
    print(f"||B|| = {for_print}")

    for_print, norm_c = vector_norm(new_matrix, round_cnt)
    print(f"||c|| = {for_print}")

    print()

    if norm_B >= 1 or check_jacobi(matrix, round_cnt):
        print("Метод нельзя использовать")
        return
    else:
        print(f"||B|| = {norm_B:.{round_cnt}f} < 1")
        print("Матрица соответствует правилам")
    print()

    if not is_seidel:
        iterations_cnt = ceil(log(round(delta * (1 - norm_B) / norm_c, round_cnt)) / log(round(norm_B, round_cnt)))
        print(
            f"\nПотребуется предположительно log({(delta * (1 - norm_B) / norm_c):.{round_cnt}f})/log({norm_B:.{round_cnt}f}) = {iterations_cnt} итераций")
    print()

    solution_and_diff = []
    if is_seidel:
        solution_and_diff = do_iterations_seidel(new_matrix, delta)
    else:
        solution_and_diff = do_iterations(new_matrix, iterations_cnt, delta)

    solutions = [round(el, 3) for el in solution_and_diff[0]]
    print(f"x = {np.abs(solutions)}")
    print(
        f"∆x = {(norm_B):.{round_cnt}f} / {(1 - norm_B):.{round_cnt}f} * {solution_and_diff[1]:.{round_cnt + 1}f} <= {(norm_B / (1 - norm_B) * solution_and_diff[1]):.{round_cnt + 2}f}")


def main():
    matrix = [[5.482, 0.358, 0.237, 0.409, 0.416],
              [0.580, 4.953, 0.467, 0.028, 0.464],
              [0.319, 0.372, 8.935, 0.520, 0.979],
              [0.043, 0.459, 0.319, 4.778, 0.126]]
    # matrix = [[5.526, 0.305, 0.887, 0.037, 0.774],
    #           [0.658, 2.453, 0.678, 0.192, 0.245],
    #           [0.398, 0.232, 4.957, 0.567, 0.343],
    #           [0.081, 0.521, 0.192, 4.988, 0.263]]

    delta = input("Введите точность вычислений: ")
    try:
        delta = float(delta)
    except:
        delta = 0.001
        print(f"Значение не распознано, будет установлено {delta}")

    print()

    print("Изначальная матрица")
    print_matrix(matrix)

    print("\nМетод Гаусса")
    gauss_method(matrix, delta)
    print("\nМетод Якоби")
    jacobi_method(matrix, delta)
    print("\nМетод Зейделя")
    jacobi_method(matrix, delta, is_seidel=True)


main()
