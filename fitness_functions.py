import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import sys


def holder_table(x, y):
    if type(x) == np.float64 or type(x) == float:
        if not -10 <= x <= 10 or not -10 <= y <= 10:
            return 10000
    f = 1 - np.sqrt(x**2 + y**2) / np.pi
    f = np.exp(np.abs(f))
    f = np.sin(x) * np.cos(y) * f
    return -np.abs(f)


def bukin6(x, y):
    if type(x) == np.float64 or type(x) == float:
        if not -15 <= x <= -5 or not -3 <= y <= 3:
            return 10000
    return 100 * np.sqrt(np.absolute(y - 0.01 * x ** 2.0)) + 0.01 * np.absolute(x + 10)


def cross_in_tray(x, y):
    if type(x) == np.float64 or type(x) == float:
        if not -10 <= x <= 10 or not -10 <= y <= 10:
            return 10000
    f = 100 - np.sqrt(x**2 + y**2) / np.pi
    f = np.sin(x) * np.sin(y) * np.exp(np.abs(f))
    f = np.abs(f) + 1
    f = -0.0001 * f**0.1
    return f


if __name__ == '__main__':
    # Holder
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(-10, 10, 0.5)
    Y = np.arange(-10, 10, 0.5)
    X, Y = np.meshgrid(X, Y)
    Z = holder_table(X, Y)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plt.show()

    # Bukin
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(-15, -5, 0.1)
    Y = np.arange(-3, 3, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = bukin6(X, Y)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plt.show()

    # Cross
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(-10, 10, 0.5)
    Y = np.arange(-10, 10, 0.5)
    X, Y = np.meshgrid(X, Y)
    Z = cross_in_tray(X, Y)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plt.show()
