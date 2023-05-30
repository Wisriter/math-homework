import numpy as np
import sympy as sp
from autograd import hessian
from scipy.optimize import minimize_scalar


# 目标函数
def objective_function(x):
    x = x.astype(float)
    fun1 = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    fun2 = (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (
            2.625 - x[0] + x[0] * x[1] ** 3) ** 2

    fun3 = (1 + (x[0] + x[1] + 1) ** 2 * (
            19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) * (
                   30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                   18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
    if fun_flag == 1:
        fun = fun1
    elif fun_flag == 2:
        fun = fun2
    elif fun_flag == 3:
        fun = fun3
    else:
        print("Please choose the correct object function.")
    return fun


# 计算梯度
def gradient(x0):
    # 定义符号变量
    x, y = sp.symbols('x y')

    # 定义表达式
    expr1 = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    expr2 = (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (
            2.625 - x + x * y ** 3) ** 2

    expr3 = (1 + (x + y + 1) ** 2 * (
            19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                    30 + (2 * x - 3 * y) ** 2 * (
                    18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))

    if fun_flag == 1:
        expr = expr1
    elif fun_flag == 2:
        expr = expr2
    elif fun_flag == 3:
        expr = expr3
    else:
        print("Please choose the correct object function.")

    # 计算梯度
    grad_expr = [sp.diff(expr, var) for var in [x, y]]

    # 计算在指定点的梯度值
    point = {x: x0[0], y: x0[1]}
    grad_values = [grad.subs(point) for grad in grad_expr]
    grad_array = np.array(grad_values)
    return grad_array


# 计算海森矩阵
def hessian_f(x0):
    # 计算Hessian矩阵函数
    hessian_matrix = hessian(objective_function)
    # 计算Hessian矩阵
    hessian_values = hessian_matrix(x0)
    return hessian_values

# def draw(x_list, y_list):
#
#
#
#     return -1


"""
梯度下降法
"""


def gradient_descent(x0, learning_rate=0.001, max_iterations=20000, tolerance=1e-3):
    x = x0
    for i in range(max_iterations):
        grad = gradient(x)
        grad = grad.astype(float)
        if np.linalg.norm(grad) < tolerance:
            break
        x = x - learning_rate * grad
    return x, i

# 动态更新学习率
def calculate_learning_rate(current_point, dk):
    # 定义符号变量，变成关于a的一元函数
    x, y, a = sp.symbols('x y a')

    # 定义x和y关于z的线性表达式
    x_expr = current_point[0] + a * dk[0]
    y_expr = current_point[1] + a * dk[1]
    # 定义二元函数
    expr1 = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    expr2 = (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (
            2.625 - x + x * y ** 3) ** 2

    expr3 = (1 + (x + y + 1) ** 2 * (
            19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                    30 + (2 * x - 3 * y) ** 2 * (
                    18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    if fun_flag == 1:
        f = expr1
    elif fun_flag == 2:
        f = expr2
    elif fun_flag == 3:
        f = expr3
    else:
        print("Please choose the correct object function.")

    # 将f表示为关于z的一元函数
    f_a = f.subs([(x, x_expr), (y, y_expr)])
    # 将符号表达式转换为可进行数值计算的函数
    f_a_func = sp.lambdify(a, f_a)
    # 使用 minimize_scalar 函数求解最小值
    result = minimize_scalar(f_a_func, bounds=(0, 1e+7))  # 区间[0, None)不行，得用一个大数作为正无穷,(0, 1e+7)合适

    # 提取最小值对应的 x 值
    minimum_a = result.x
    return minimum_a


"""
1. 最速下降
"""


def steepest_descent(x0, max_iterations, tolerance):
    current_point = x0
    x_list = []
    y_list = []
    i = 0
    while i < max_iterations:
        grad = gradient(current_point)
        dk = -grad
        dk = dk.astype(float)
        if np.linalg.norm(dk) < tolerance:
            break
        else:
            lr = calculate_learning_rate(current_point, dk)  # 动态计算学习率
            x_list.append(current_point[0])
            y_list.append(current_point[1])
            current_point = current_point + lr * dk
            print(current_point, "\t", objective_function(current_point))
            i += 1
    if i < max_iterations:
        print(f"最速下降法第{i}次迭代：目标函数值为：{objective_function(current_point)}，在{current_point}处取得\n")
    else:
        print("最速下降法无法收敛")
    return x_list, y_list


"""
2. 阻尼牛顿
"""


def damped_newton(x0, max_iterations, tolerance):
    current_point = x0
    x_list = []
    y_list = []
    for i in range(max_iterations):
        grad = gradient(current_point).astype(float)
        hessian = hessian_f(current_point)
        dk = -np.linalg.solve(hessian, grad)
        if np.linalg.norm(dk) < tolerance:
            break
        else:
            lr = calculate_learning_rate(current_point, dk)  # 动态计算学习率
            x_list.append(current_point[0])
            y_list.append(current_point[1])
            current_point = current_point + lr * dk
            print(current_point, "\t", objective_function(current_point))
            i += 1

    if i < max_iterations:
        print(f"阻尼牛顿法第{i}次迭代：目标函数值为：{objective_function(current_point)}，在{current_point}处取得\n")
    else:
        print("阻尼牛顿法无法收敛")
    return x_list, y_list


"""
3. BFGS方法
"""


# 线搜索函数，用于确定步长 alpha
def line_search(f, x, p, max_iterations=100, alpha=1.0, c=0.5, rho=0.5):
    for i in range(max_iterations):
        if f(x + alpha * p) <= f(x) + c * alpha * np.dot(gradient(x), p):
            return alpha
        else:
            alpha *= rho
    return alpha


def bfgs_method(x0, max_iterations=1000, tolerance=1e-6):
    x = x0
    x_list = []
    y_list = []
    H = np.eye(len(x0))  # 初始化 Hessian 矩阵为单位矩阵

    for i in range(max_iterations):
        grad = gradient(x)
        grad = grad.astype(float)
        if np.linalg.norm(grad) < tolerance:
            break

        p = -np.dot(H, grad)  # 计算搜索方向

        alpha = line_search(objective_function, x, p)  # 使用线搜索确定步长

        s = alpha * p

        x_new = x + s

        y = gradient(x_new) - grad
        rho = 1 / np.dot(y, s)

        H = (np.eye(len(x)) - rho * np.outer(s, y)) @ H @ (np.eye(len(x)) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x_list.append(x[0])
        y_list.append(x[1])
        x = x_new
    if i < max_iterations:
        print(f"BFGS方法第{i}次迭代：目标函数值为：{objective_function(x)}，在{x}处取得\n")
    else:
        print("BFGS方法无法收敛")
    return x_list, y_list


if __name__ == '__main__':
    """
    3道题的解：               迭代初值：
    （1）：(1, 1);           [0.9,1.1]
    （2）：(3, 0.5);         [3.2, 0.7]
    （3）：(0, -1)           [0, -0.9]  # 第三题初值需要取得很接近解，不然梯度会很大，无法收敛
    """
    dict = {1: np.array([0.9, 1.1]),
            2: np.array([3.2, 0.7]),
            3: np.array([0, -0.9])}
    fun_flag = 1
    x0 = dict[fun_flag]  # 初始值

    x_list1, y_list1 = steepest_descent(x0, max_iterations=1000, tolerance=1e-6)  # tolerance=1e-6
    x_list2, y_list2 = damped_newton(x0, max_iterations=1000, tolerance=1e-6)
    x_list3, y_list3 = bfgs_method(x0, max_iterations=1000, tolerance=1e-6)
