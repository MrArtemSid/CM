from prettytable import PrettyTable
import sympy as sp
import matplotlib.pyplot as plt
from bisect import bisect_left
import numpy as np
from math import ceil, log


def paint(x_values, y_values, interpolated_x=[], interpolated_y=[], name=""):
    plt.figure()
    plt.scatter(x_values, y_values, color='blue', marker='o', label="Известные решения")
    if name:
        plt.plot(interpolated_x, interpolated_y, label=f"Полином {name}", linestyle="--")
    plt.title("f(x)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    plt.show()


def generate_polynoms(x_values, y_values):
    polynoms = []
    for i in range(len(x_values)):
        curr_polynom = 1
        for j in range(len(x_values)):
            if i != j:  # исключаем текущий индекс
                curr_polynom *= (x_values[i] - x_values[j])
        inverse_p = 1 / curr_polynom
        polynoms.append([curr_polynom, inverse_p, y_values[i] * inverse_p])
    return polynoms


def generate_y_L(x_values, interpolated_x, polynoms, precision):
    interpolated_y = []
    f_obj = 0
    x_sym, one = sp.symbols("x 1")
    # Единица(one) нужна для того чтобы в L(x) первый аргумент сразу не перемножался с первой парой, иначе sympy сразу перемножит их

    print()
    print(f"L(x) = ", end="")

    for i in range(len(x_values)):
        curr_f_obj = polynoms[i][-1] * one
        for j in range(len(x_values)):
            if i != j:
                curr_f_obj *= (x_sym - x_values[j])
        f_obj += curr_f_obj
        print(curr_f_obj.evalf(precision), "+" if i < len(x_values) - 1 else "")
    print()

    f = sp.lambdify((x_sym, one), f_obj)
    for curr_x in interpolated_x:
        interpolated_y.append(round(f(curr_x, 1), 3))

    return interpolated_y, f


def print_polynom_table(polynoms, precision):
    table = PrettyTable()
    table.field_names = ["p", "Знамен", "1/знам", "к.п.*f"]
    for i, polynom in enumerate(polynoms):
        table.add_row([f"p{i}"] + [f"{el:.{precision}f}" for el in polynom])
    print("Базисные полиномы", table, sep="\n")


def generate_diff_table(x, y, is_splitted=False):
    n = len(x)
    table = [[x[i], y[i]] + [0] * (n - i - 1) for i in range(n)]

    # Внешний цикл для вычисления разностей разного порядка (от 2 до n)
    for mov in range(2, n + 1):
        # Внутренний цикл для заполнения соответствующих разностей в таблице
        for i in range(n - mov + 1):
            # Вычисление разности значений y для текущего порядка (mov - 1)
            diff_y = table[i + 1][mov - 1] - table[i][mov - 1]

            if is_splitted:
                # Для разделенных разностей: делим разность y на разность x
                diff_x = table[i + mov - 1][0] - table[i][0]
                table[i][mov] = diff_y / diff_x
            else:
                # Для обычных разностей: просто сохраняем разность y
                table[i][mov] = diff_y

    return table


def print_diff_table(table, precision):
    # Настройка и вывод таблицы с использованием PrettyTable
    columns = ["x", "y"] + [str(i) for i in range(1, len(table))]
    pretty_table = PrettyTable(columns)

    for row in table:
        tmp = [f"{el:.{precision}f}" for el in row]
        # Заполняем пустыми строками строки, где нет значений конечных разностей
        while len(tmp) < len(columns):
            tmp.append("")
        pretty_table.add_row(tmp)
    print(pretty_table)


def print_values(x_values, y_values, precision):
    columns = ["x", "y"]
    xy_table = PrettyTable(columns)
    for i in range(len(x_values)):
        xy_table.add_row([f"{x_values[i]:.{precision}f}", f"{y_values[i]:.{precision}f}"])
    print(xy_table)


def generate_y_N(interpolated_x, splitted_table):
    interpolated_y = []
    x_sym, one = sp.symbols("x 1")
    # Единица(one) нужна для того чтобы первый аргумент сразу не перемножался с первой парой, иначе sympy сразу перемножит их
    f_obj = 0  # Начальное значение функции — первая разность (нулевого порядка)
    n = len(splitted_table)

    # Вывод формулы
    print(f"N4(x) = ", end="")

    # Генерация формулы
    product_term = 1 * one
    for i in range(n):
        if i != 0:
            product_term *= (x_sym - splitted_table[i - 1][0])
        f_obj += splitted_table[0][i + 1] * product_term
        print(splitted_table[0][i + 1] * product_term, "+" if i < n - 1 else "")

    # Компилируем функцию f из символической формулы
    f = sp.lambdify((x_sym, one), f_obj)

    # Вычисление значений для interpolated_x
    for curr_x in interpolated_x:
        interpolated_y.append(round(f(curr_x, 1), 3))

    return interpolated_y, f


def linear_spline(x_values, y_values, interpolated_x, precision):
    interpolated_y = []

    spline_equations_table = PrettyTable()
    spline_equations_table.field_names = ["Уравнения", "Интервалы"]

    pretty_table1 = PrettyTable()
    pretty_table1.field_names = ["Уравнения"]

    pretty_table2 = PrettyTable()
    pretty_table2.field_names = ["a", "b"]

    phi_table = PrettyTable()
    phi_table.field_names = ["Уравнения", "Интервалы"]

    for i in range(len(x_values) - 1):
        # Вычисляем коэффициенты a и b
        a = (y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i])
        b = y_values[i] - a * x_values[i]

        a = f"{a:.{precision}f}"
        b = f"{b:.{precision}f}"

        # Добавляем строки в таблицы
        pretty_table1.add_row([f"{x_values[i]}a{i + 1} + b{i + 1} = {y_values[i]}"])
        pretty_table1.add_row([f"{x_values[i + 1]}a{i + 1} + b{i + 1} = {y_values[i + 1]}"])
        spline_equations_table.add_row([f"a{i + 1}x + b{i + 1}", f"{x_values[i]} <= x <= {x_values[i + 1]}"])
        pretty_table2.add_row([f"a{i + 1} = {a}", f"b{i + 1} = {b}"])
        phi_table.add_row([f"{a}x + {b}", f"{x_values[i]} <= x <= {x_values[i + 1]}"])

    # Проходим по всем точкам interpolated_x
    for curr_x in interpolated_x:
        i = bisect_left(x_values, curr_x) - 1
        if x_values[i] <= curr_x <= x_values[i + 1]:
            # Вычисляем линейное значение y на этом интервале
            a = (y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i])
            b = y_values[i] - a * x_values[i]
            value = a * curr_x + b
            interpolated_y.append(value)
        else:
            # Если точка вне диапазона интерполяции, используем ближайшую точку
            if curr_x <= x_values[0]:
                interpolated_y.append(y_values[0])
            else:
                interpolated_y.append(y_values[-1])

    print(spline_equations_table)
    print("Система")
    print(pretty_table1)
    print("Решение системы")
    print(pretty_table2)
    print("Линейный сплайн")
    print(phi_table)

    return interpolated_y


def square_spline(x_values, y_values, interpolated_x, precision):
    interpolated_values = []

    equations_table = PrettyTable()
    equations_table.field_names = ["Уравнения", "Интервалы"]

    equations_table.add_row(["a1x^2 + b1x + c1", f"x∈[{x_values[0]};{x_values[len(x_values) // 2]}]"])
    equations_table.add_row(["a2x^2 + b2x + c2", f"x∈[{x_values[len(x_values) // 2]};{x_values[-1]}]"])

    system_equations_table = PrettyTable()
    system_equations_table.field_names = ["Уравнения"]

    for index in range(len(x_values)):
        segment = int(index // (len(x_values) / 2)) + 1
        system_equations_table.add_row(
            [f"{x_values[index]}^2a{segment} + {x_values[index]}b{segment} + c{segment} = {y_values[index]}"])
        if index == len(x_values) // 2:
            segment += 1
            system_equations_table.add_row(
                [f"{x_values[index]}^2a{segment} + {x_values[index]}b{segment} + c{segment} = {y_values[index]}"])

    # Матрица коэффициентов (A) и вектор правых частей (B)
    coefficients_matrix = np.array([
        [x_values[0] ** 2, x_values[0], 1, 0, 0, 0],
        [x_values[1] ** 2, x_values[1], 1, 0, 0, 0],
        [x_values[2] ** 2, x_values[2], 1, 0, 0, 0],
        [0, 0, 0, x_values[2] ** 2, x_values[2], 1],
        [0, 0, 0, x_values[3] ** 2, x_values[3], 1],
        [0, 0, 0, x_values[4] ** 2, x_values[4], 1]
    ])

    constants_vector = np.array([y_values[0], y_values[1], y_values[2], y_values[2], y_values[3], y_values[4]])

    # Решение системы линейных уравнений
    solution = np.linalg.solve(coefficients_matrix, constants_vector)
    print(equations_table)
    print("Система")
    print(system_equations_table)

    a1, b1, c1, a2, b2, c2 = solution
    coefficients_table = PrettyTable()
    coefficients_table.field_names = ["a", "b", "c"]
    coefficients_table.add_row([f"a1 = {a1:.{precision}f}", f"b1 = {b1:.{precision}f}", f"c1 = {c1:.{precision}f}"])
    coefficients_table.add_row([f"a2 = {a2:.{precision}f}", f"b2 = {b2:.{precision}f}", f"c2 = {c2:.{precision}f}"])

    print("Решение системы")
    print(coefficients_table)

    print("Квадратичный сплайн")
    spline_table = PrettyTable()
    spline_table.field_names = ["Уравнения", "Интервалы"]

    x_symbol = sp.symbols("x")
    spline_first_obj = a1 * x_symbol ** 2 + b1 * x_symbol + c1
    spline_second_obj = a2 * x_symbol ** 2 + b2 * x_symbol + c2

    spline_first = sp.lambdify((x_symbol), spline_first_obj)
    spline_second = sp.lambdify((x_symbol), spline_second_obj)

    spline_table.add_row([spline_first_obj.evalf(precision), f"x∈[{x_values[0]};{x_values[len(x_values) // 2]}]"])
    spline_table.add_row([spline_second_obj.evalf(precision), f"x∈[{x_values[len(x_values) // 2]};{x_values[-1]}]"])

    print(spline_table)

    for point in interpolated_x:
        if x_values[0] <= point <= x_values[len(x_values) // 2]:
            interpolated_values.append(spline_first(point))
        else:
            interpolated_values.append(spline_second(point))
    return interpolated_values


def print_comparing(x_values, y_values, interpolated_x, interpolated_y_L, interpolated_y_N, interpolated_y_S,
                    interpolated_y_SQ):
    plt.figure()
    plt.plot(interpolated_x, interpolated_y_L, label="Полином Лагранжа", linestyle="--")
    plt.plot(interpolated_x, interpolated_y_N, label="Полином Ньютона", linestyle="-.")
    plt.plot(interpolated_x, interpolated_y_S, label="Линейный сплайн", linestyle=":")
    plt.plot(interpolated_x, interpolated_y_SQ, label="Квадратичный сплайн", linestyle=":")
    plt.scatter(x_values, y_values, color='blue', label="Известные решения", zorder=5)
    plt.title("Сравнение методов интерполяции")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    x_values = [2.119, 3.618, 5.342, 7.859, 8.934]
    y_values = [0.605, 0.718, 0.105, 2.157, 3.431]

    #x = [0.083, 0.472, 1.347, 2.117, 2.947]
    #y = [-2.132, -2.013, -1.613, -0.842, 2.973]

    #x = [0.134, 0.561, 1.341, 2.291, 6.913]
    #y = [2.156, 3.348, 3.611, 4.112, 4.171]

    eps = float(input("Введите погрешность вычислений: "))
    precision = ceil(abs(log(eps, 10)))

    step = float(input("Введите шаг интерполяции (например 0.1): "))
    x_keyboard = float(input("Введите x: "))

    scale = 1 / step  # Определяем масштаб в зависимости от шага

    interpolated_x = sorted(
        {x / scale for x in range(int(min(x_values) * scale), int(max(x_values) * scale), 1)}
        | set(x_values)
        | {x_keyboard}
    )

    paint(x_values, y_values)
    print("Изначальные значения")
    print_values(x_values, y_values, precision)

    polynoms = generate_polynoms(x_values, y_values)
    print_polynom_table(polynoms, precision)

    x_keyboard_i = interpolated_x.index(x_keyboard)
    interpolated_y_L, L = generate_y_L(x_values, interpolated_x, polynoms, precision)
    print("\nИнтерполяция методом Лагранжа:")
    print_values(interpolated_x, interpolated_y_L, precision)
    paint(x_values, y_values, interpolated_x, interpolated_y_L, "Лагранжа")

    print("Таблица конечных разностей")
    table = generate_diff_table(x_values, y_values)
    print_diff_table(table, precision)

    print("Таблица разделенных разностей")
    splitted_table = generate_diff_table(x_values, y_values, is_splitted=True)
    print_diff_table(splitted_table, precision)

    interpolated_y_N, N = generate_y_N(interpolated_x, splitted_table)
    print("\nИнтерполяция методом Ньютона:")
    print_values(interpolated_x, interpolated_y_N, precision)
    paint(x_values, y_values, interpolated_x, interpolated_y_N, "Ньютона")

    x1 = x_values[1]
    x2 = x_values[2]

    print()
    print(f"Вычислим значение полинома Лагранжа в точке x1+x2: L4({x1} + {x2}) = {L(x1 + x2, 1):.{precision}f}")
    print(f"Вычислим значение полинома Ньютона в точке x1+x2: N4({x1} + {x2}) = {N(x1 + x2, 1):.{precision}f}")
    print()

    interpolated_y_S = linear_spline(x_values, y_values, interpolated_x, precision)
    print("\nИнтерполяция Линейным сплайном:")
    print_values(interpolated_x, interpolated_y_S, precision)
    paint(x_values, y_values, interpolated_x, interpolated_y_S, "Линейный сплайн")

    interpolated_y_SQ = square_spline(x_values, y_values, interpolated_x, precision)
    print("\nИнтерполяция Квадратичным сплайном:")
    print_values(interpolated_x, interpolated_y_SQ, precision)
    paint(x_values, y_values, interpolated_x, interpolated_y_SQ, "Квадратичный сплайн")

    print(f"x: {x_keyboard}")
    print(f"Значение вычисленное через полином Лагранжа: {interpolated_y_L[x_keyboard_i]:.{precision}f}")
    print(f"Значение вычисленное через полином Ньютона: {interpolated_y_N[x_keyboard_i]:.{precision}f}")
    print(f"Значение вычисленное через линейный сплайн: {interpolated_y_S[x_keyboard_i]:.{precision}f}")
    print(f"Значение вычисленное через квадратичный сплайн: {interpolated_y_SQ[x_keyboard_i]:.{precision}f}")

    # Построение графиков для сравнения
    print_comparing(x_values, y_values, interpolated_x, interpolated_y_L, interpolated_y_N, interpolated_y_S,
                    interpolated_y_SQ)


main()
