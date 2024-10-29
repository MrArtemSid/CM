import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import inv, norm
from prettytable import PrettyTable
from math import ceil, log
from warnings import filterwarnings

def find_sign_changes(x_values, y1_values, y2_values):
    intervals = []
    for i in range(len(y1_values) - 1):
        if (y1_values[i] - y2_values[i]) * (y1_values[i + 1] - y2_values[i + 1]) < 0:
            intervals.append((x_values[i], x_values[i + 1]))
    return intervals


def paint_f(f1_obj, f2_obj, x, label1, min_value, max_value):
    f1 = sp.lambdify(x, f1_obj)
    f2 = sp.lambdify(x, f2_obj)

    x_values = np.linspace(min_value, max_value, int((max_value - min_value) * 100))
    y1_values = [f1(x_val) for x_val in x_values]
    y2_values = [f2(x_val) for x_val in x_values]

    plt.plot(x_values, y1_values, label=f'{label1} = {f1_obj}')
    plt.plot(x_values, y2_values, label=f'{label1} = {f2_obj}')
    plt.axhline(0, color='black', linewidth=0.5)  # Ось x
    plt.axvline(0, color='black', linewidth=0.5)  # Ось y
    plt.xticks(np.arange(min_value, max_value + 0.1, 0.1))  # Метки по оси x
    plt.grid(True)
    plt.title(f'График функции {label1}')
    plt.xlabel('x1')
    plt.ylabel(label1)

    intervals = find_sign_changes(x_values, y1_values, y2_values)
    roots = []
    cross_f = sp.lambdify(x, f1_obj - f2_obj)
    for interval in intervals:
        root = fsolve(cross_f, (interval[0] + interval[1]) / 2)
        roots.append(root[0])

    intersection_points_x = np.array(roots)
    intersection_points_y = [f1(x_val) for x_val in intersection_points_x]

    plt.scatter(intersection_points_x, intersection_points_y, color='red', zorder=5, label='Точки пересечения')

    plt.legend()
    plt.show()


def newton_method_system(f1_obj, f2_obj, x1, x2, initial_guess, eps=1e-6, max_iter=100):
    precision = ceil(abs(log(eps, 10))) + 1

    jacobian_matrix_obj = sp.Matrix([
        [sp.diff(f1_obj, x1), sp.diff(f1_obj, x2)],
        [sp.diff(f2_obj, x1), sp.diff(f2_obj, x2)]
    ])

    J_table = PrettyTable()
    J_table.field_names = ["∂f/∂x1", "∂f/∂x2"]

    J_table.add_row([jacobian_matrix_obj[0, 0], jacobian_matrix_obj[0, 1]])
    J_table.add_row([jacobian_matrix_obj[1, 0], jacobian_matrix_obj[1, 1]])

    print(J_table)

    f1 = sp.lambdify((x1, x2), f1_obj)
    f2 = sp.lambdify((x1, x2), f2_obj)
    jacobian_matrix = sp.lambdify((x1, x2), jacobian_matrix_obj)

    x = np.array(initial_guess)
    prev = x.copy()

    ans_table = PrettyTable()
    ans_table.field_names = ["k", "x1", "x2", "Норма"]
    ans_table.add_row([0, f"{x[0]:.{precision}f}", f"{x[1]:.{precision}f}", "-"])

    for k in range(1, max_iter):
        j_curr = jacobian_matrix(*x)

        inverse_J = inv(j_curr)

        x -= inverse_J @ [f1(*x), f2(*x)]

        delta = norm(x - prev)
        prev = x.copy()

        ans_table.add_row([k, f"{x[0]:.{precision}f}", f"{x[1]:.{precision}f}", f"{delta:.{precision}f}"])
        if delta < eps:
            break
    else:
        print(f"Решение не сошлось за {iter} итераций")

    print(ans_table)
    return x


def gradient_descent(grad_f_x1, grad_f_x2, initial_guess, eps=1e-6, max_iter=100):
    precision = ceil(abs(log(eps, 10))) + 1

    x_start = np.array(initial_guess, dtype=float)

    alpha_step = 1e-3
    alpha = alpha_step

    best_k = max_iter + 1

    while alpha <= 1:
        x = x_start.copy()
        prev = x.copy()

        ans_table = PrettyTable()
        ans_table.field_names = ["k", "x1", "x2", "Норма"]
        ans_table.add_row([0, f"{x[0]:.{precision}f}", f"{x[1]:.{precision}f}", "-"])

        for k in range(1, max_iter):
            grad = np.array([grad_f_x1(x[0], x[1]), grad_f_x2(x[0], x[1])])

            x -= alpha * grad

            delta = norm(x - prev)
            prev = x.copy()

            ans_table.add_row([k, f"{x[0]:.{precision}f}", f"{x[1]:.{precision}f}", f"{delta:.{precision}f}"])

            if delta < eps:
                break

        alpha += alpha_step

        if k < best_k:
            best_k, best_ans_table, best_alpha, best_x = k, ans_table, alpha, x

    alpha_table = PrettyTable()
    alpha_table.field_names = ["alpha"]
    alpha_table.add_row([best_alpha])

    print(alpha_table, best_ans_table, sep="\n")
    return best_x


def gradient_descent_optimized(grad_f, grad_f_x1, grad_f_x2, initial_guess, eps=1e-6, max_iter=100):
    def ternary_search_for_alpha(x, grad, alpha_left=0, alpha_right=1, max_iter=100):
        def ternary_f(alpha, x, grad):
            new_x = x - alpha * grad
            return grad_f(*new_x)

        left, right = alpha_left, alpha_right
        for _ in range(max_iter):
            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3
            if ternary_f(mid1, x, grad) > ternary_f(mid2, x, grad):
                left = mid1
            else:
                right = mid2
        return (left + right) / 2

    precision = ceil(abs(log(eps, 10))) + 1

    x = np.array(initial_guess, dtype=float)
    prev = x.copy()

    ans_table = PrettyTable()
    ans_table.field_names = ["k", "x1", "x2", "alpha", "Норма"]
    ans_table.add_row([0, f"{x[0]:.{precision}f}", f"{x[1]:.{precision}f}", "-", "-"])

    for k in range(1, max_iter):
        grad = np.array([grad_f_x1(x[0], x[1]), grad_f_x2(x[0], x[1])])
        alpha = ternary_search_for_alpha(x, grad)

        x -= alpha * grad

        delta = norm(x - prev)
        prev = x.copy()

        ans_table.add_row(
            [k, f"{x[0]:.{precision}f}", f"{x[1]:.{precision}f}", f"{alpha:.{precision}f}", f"{delta:.{precision}f}"])

        # Проверка условия сходимости
        if delta < eps:
            break

    print(ans_table)
    return x


def main():
    # Для исключения предупреждений выхода аргумента за пределы области определения
    filterwarnings("ignore", category=RuntimeWarning)

    min_value, max_value = -10, 10
    x1, x2 = sp.symbols("x1 x2")

    eps = float(input("Введите погрешность: "))
    precision = ceil(abs(log(eps, 10)))

    f1_obj = sp.sin(x1 + x2) - 1.5 * x1 - 0.1
    f2_obj = 3 * x1 ** 2 + x2 ** 2 - 1

    print(f"f1 = {f1_obj}")
    print(f"f2 = {f2_obj}")

    x2_1_obj = sp.solve(f1_obj, x2)[0]
    x2_2_obj = sp.solve(f2_obj, x2)[1]

    print("x2 =", x2_1_obj)
    print("x2 =", x2_2_obj)

    paint_f(x2_1_obj, x2_2_obj, x1, "x2", min_value, max_value)

    grad_f_obj = (f1_obj ** 2 + f2_obj ** 2).nsimplify()
    grad_f_x1_obj = sp.diff(grad_f_obj, x1)
    grad_f_x2_obj = sp.diff(grad_f_obj, x2)

    grad_f = sp.lambdify((x1, x2), grad_f_obj)
    grad_f_x1 = sp.lambdify((x1, x2), grad_f_x1_obj)
    grad_f_x2 = sp.lambdify((x1, x2), grad_f_x2_obj)

    # Задаем начальное приближение
    initial_guess = [0.49, 0.51]

    solution_newton = newton_method_system(f1_obj, f2_obj, x1, x2, initial_guess, eps)
    print("Найденное решение методом Ньютона:", solution_newton.round(precision))
    print()

    print(f"Ф = {grad_f_obj}")
    print(f"Ф_x1 = {grad_f_x1_obj}")
    print(f"Ф_x2 = {grad_f_x2_obj}")

    solution_gradient = gradient_descent(grad_f_x1, grad_f_x2, initial_guess, eps)
    print("Найденное решение методом наискорейшего спуска:", solution_gradient.round(precision))

    solution_gradient = gradient_descent_optimized(grad_f, grad_f_x1, grad_f_x2, initial_guess, eps)
    print("Найденное решение методом оптимизированного наискорейшего спуска:", solution_gradient.round(precision))


main()
