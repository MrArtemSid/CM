import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from math import ceil, log


def find_sign_changes(x_values, y_values):
    intervals = []
    for i in range(len(y_values) - 1):
        if y_values[i] * y_values[i + 1] < 0:
            intervals.append((x_values[i], x_values[i + 1]))
    return intervals


def paint_f(f_obj, x, label, min_value, max_value, show_cross_ox=False):
    f = sp.lambdify(x, f_obj)
    x_values = np.linspace(min_value, max_value, (max_value - min_value) * 10)
    y_values = [f(x_val) for x_val in x_values]

    # Построение графика
    plt.plot(x_values, y_values, label=f'{label} = {f_obj}')
    plt.axhline(0, color='black', linewidth=0.5)  # Ось x
    plt.axvline(0, color='black', linewidth=0.5)  # Ось y
    plt.grid(True)
    plt.title(f'График функции {label}')
    plt.xlabel('x')
    plt.ylabel(label)
    plt.legend()

    if show_cross_ox:
        intervals = find_sign_changes(x_values, y_values)
        for interval in intervals:
            mid = (interval[0] + interval[1]) / 2
            plt.plot(mid, 0, 'ro')

    plt.show()
    return x_values, y_values


def print_iteration_header():
    print(f"{'k':<5}{'xk':<20}{'f(xk)':<20}{'||x_k - x_(k-1)||':<20}")


def iterate(f, func, x0, eps, max_iter=100):
    precision = ceil(abs(log(eps, 10))) + 1

    print(f"Начальное приближение: x0 = {x0:.{precision}f}")
    print_iteration_header()
    results = [[x0, f(x0)]]

    print(f"{0:<5}{x0:<20.{precision}f}{results[0][1]:<20.{precision}f}{'-'}")
    for k in range(1, max_iter):
        prev_x = results[-1][0]
        curr_x = func(prev_x)
        diff = abs(curr_x - prev_x)
        results.append([curr_x, f(curr_x)])

        print(f"{k:<5}{curr_x:<20.{precision}f}{results[-1][1]:<20.{precision}f}{diff:<20.{precision}f}")
        if diff <= eps:
            print(f"\nРешение достигнуто на итерации {k} \nx = {curr_x:.{precision}f}\nПогрешность = {eps}")
            return curr_x
    print(f"Не удалось найти решение за {max_iter} итераций")
    return None


def iteration(x_values, y_values, f_obj, x, min_value, max_value):
    eps = 1e-3
    precision = ceil(abs(log(eps, 10))) + 1

    Q = max([abs(y) for y in y_values])
    print(f"Q = max(|f'(x)|) = {Q:.{precision}f}")
    is_positive = 1 if Q in y_values else -1
    t = ceil(Q / 2) * is_positive
    print(f"|t| >= Q / 2 = {Q:.{precision}f} / 2 ~= {t:.{precision}f}")

    phi_obj = x - f_obj / t
    diff_phi_obj = phi_obj.diff()
    diff_phi = sp.lambdify(x, diff_phi_obj)
    print(f"Ф(x) = {phi_obj}")
    print(f"Ф'(x) = {diff_phi_obj}")

    paint_f(diff_phi_obj, x, "Ф'(x)", min_value, max_value)

    y_values_for_phi_diff = [diff_phi(x) for x in x_values]
    q = max([abs(y) for y in y_values_for_phi_diff])
    print(f"q = max(|Ф'(x)|) = {q:.{precision}f} < 1")
    print(f"Условие сходимости {'не' if q >= 1 else ''} выполнено")
    if q >= 1:
        return

    phi = sp.lambdify(x, phi_obj)

    try:
        ind = y_values_for_phi_diff.index(q)
    except:
        ind = y_values_for_phi_diff.index(-q)

    x0 = x_values[ind]

    print(f"x0 = {x0:.{precision}f}")
    iterate(sp.lambdify(x, f_obj), phi, x0, eps=1e-3)


def iteration_newton(x_values, x, f_obj):
    f = sp.lambdify(x, f_obj)

    diff_f_obj = f_obj.diff()
    diff_f = sp.lambdify(x, diff_f_obj)

    twodiff_f_obj = diff_f_obj.diff()
    twodiff_f = sp.lambdify(x, twodiff_f_obj)

    print(f"f(x) = {f_obj}")
    print(f"f'(x) = {diff_f_obj}")
    print(f"f''(x) = {twodiff_f_obj}")

    for curr_x in x_values:
        if f(curr_x) * twodiff_f(curr_x) > 0:
            print(f"x0 = {curr_x:.7f}, так как f({curr_x:.7f}) * f''({curr_x:.7f}) > 0")
            break

    func_obj = x - f_obj / diff_f_obj
    func = sp.lambdify(x, func_obj)

    iterate(f, func, curr_x, eps=1e-6, max_iter=100)


def iteration_steffensen(x_values, f_obj, x, x0=None, eps=1e-6):
    f = sp.lambdify(x, f_obj)

    if x0 is None:
        x0 = x_values[0]

    def phi(xk):
        f_xk = f(xk)
        return xk - (f_xk ** 2) / (f(xk + f_xk) - f_xk)

    iterate(f, phi, x0, eps)


def main():
    x = sp.symbols('x')
    f_obj = 0.5 * sp.exp(-x ** 2) + x * sp.cos(x)
    #f_obj = 2 * x ** 2 - x ** 3 - sp.exp(x)

    min_value, max_value = 1, 2

    print(f"f(x) = {f_obj}")
    paint_f(f_obj, x, "f(x)", min_value, max_value, show_cross_ox=True)

    diff_f_obj = f_obj.diff()
    print(f"f'(x) = {diff_f_obj}")
    x_values, y_values = paint_f(diff_f_obj, x, "f'(x)", min_value, max_value)

    print("\nИтерационный метод")
    iteration(x_values, y_values, f_obj, x, min_value, max_value)

    print("\nМетод Ньютона")
    iteration_newton(x_values, x, f_obj)

    print("\nМетод Стеффенсена")
    iteration_steffensen(x_values, f_obj, x)

    print()
    eps = float(input("Введите точность вычислений ε: "))
    x0 = float(input("Введите начальное приближение x0: "))
    iteration_steffensen(x_values, f_obj, x, x0, eps)


main()
