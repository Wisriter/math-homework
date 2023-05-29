"""
需要在objective_function()和gradient()两处地方选择计算哪个函数，即第20行和37行
"""
import numpy as np
import sympy as sp
from autograd import hessian
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar


def objective_function(x):
    x = x.astype(float)
    fun1 = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    fun2 = (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (
            2.625 - x[0] + x[0] * x[1] ** 3) ** 2

    fun3 = (1 + (x[0] + x[1] + 1) ** 2 * (
            19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) * (
                   30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                   18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
    return fun1


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

    expr = expr1

    # 计算梯度
    grad_expr = [sp.diff(expr, var) for var in [x, y]]

    # 计算在指定点的梯度值
    point = {x: x0[0], y: x0[1]}
    grad_values = [grad.subs(point) for grad in grad_expr]
    grad_array = np.array(grad_values)
    return grad_array


def hessian_f(x0):
    # 计算Hessian矩阵函数
    hessian_matrix = hessian(objective_function)
    # 计算Hessian矩阵
    hessian_values = hessian_matrix(x0)
    return hessian_values


def get_min_point():
    initial_guess = [1, 2]
    result = minimize(objective_function(), initial_guess, method='BFGS', tol=1e-6)
    min_point = result.x
    return min_point


"""
1. 最速下降
这个方法效果最差，需要针对具体的目标函数和初值，选取合适的学习率和tolerance
"""


def steepest_descent(x0, max_iterations=1000, tolerance=1e-7):
    current_point = x0
    i = 0
    while i < max_iterations:
        grad = gradient(current_point)
        dk = -grad
        dk = dk.astype(float)
        if np.linalg.norm(dk) < tolerance:
            break
        else:
            lr = calculate_learning_rate01(current_point, dk)  # 动态计算学习率
            current_point = current_point + lr * dk
            print(current_point, "\t", objective_function(current_point))
            i += 1
    return current_point, i


def calculate_learning_rate01(current_point, dk):
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
    f = expr1

    # 将f表示为关于z的一元函数
    f_a = f.subs([(x, x_expr), (y, y_expr)])
    # 将符号表达式转换为可进行数值计算的函数
    f_a_func = sp.lambdify(a, f_a)
    # 使用 minimize_scalar 函数求解最小值
    result = minimize_scalar(f_a_func, bounds=(0, 1e+8))  # 区间[0, None)不行，得用一个大数作为正无穷

    # 提取最小值对应的 x 值
    minimum_a = result.x
    return minimum_a


def calculate_learning_rate02(grad, x):
    # 动态计算学习率
    numerator = np.dot(grad, grad)
    denominator = np.dot(grad, np.dot(hessian_f(x), grad))
    learning_rate = numerator / denominator
    return learning_rate


"""
梯度下降
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


"""
2. 阻尼牛顿
"""


def damped_newton(f, x0, max_iterations=1000, tolerance=1e-6):
    x = x0
    for i in range(max_iterations):
        grad = gradient(x)
        hessian = hessian_f(x)
        grad = grad.astype(float)
        if np.linalg.norm(grad) < tolerance:
            break
        delta_x = np.linalg.solve(hessian, grad)
        damping = 1.0
        while f(x - damping * delta_x) > f(x):
            damping *= 0.5
        x = x - damping * delta_x
    return x, i


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


def bfgs_method(f, x0, max_iterations=1000, tolerance=1e-6):
    x = x0
    H = np.eye(len(x0))  # 初始化 Hessian 矩阵为单位矩阵

    for i in range(max_iterations):
        grad = gradient(x)
        grad = grad.astype(float)
        if np.linalg.norm(grad) < tolerance:
            break

        p = -np.dot(H, grad)  # 计算搜索方向

        alpha = line_search(f, x, p)  # 使用线搜索确定步长

        s = alpha * p
        x_new = x + s

        y = gradient(x_new) - grad
        rho = 1 / np.dot(y, s)

        H = (np.eye(len(x)) - rho * np.outer(s, y)) @ H @ (np.eye(len(x)) - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x = x_new

    return x, i


if __name__ == '__main__':
    """
    3道题的解：
    （1）：(1, 1);
    （2）：(3, 0.5);
    （3）：(1.2, 0.8)
    """
    x0 = np.array([0.9, 1.1])  # 初始值

    xy_1, i = steepest_descent(x0)
    print(f"最速下降法第{i}次迭代：目标函数值为：{objective_function(xy_1)}，在{xy_1}处取得\n")

    # xy_2, j = damped_newton(objective_function, x0)
    # print(f"牛顿阻尼法第{j}次迭代：目标函数值为：{objective_function(xy_2)}，在{xy_2}处取得\n")
    #
    # xy_3, k = bfgs_method(objective_function, x0)
    # print(f"BFGS方法第{k}次迭代：目标函数值为：{objective_function(xy_3)}，在{xy_3}处取得\n")
