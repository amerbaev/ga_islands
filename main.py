from itertools import combinations

import numpy as np
from tqdm import tqdm
import copy
import deap.tools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from fitness_functions import bukin6, holder_table, cross_in_tray

# функция для оптимизации
# func = lambda x, y: (x - 5) ** 2 + (y - 1) ** 2
# func = lambda x, y: (x - 5) ** 2 + (y - 1) ** 2
# func = lambda x, y: (x - 5) ** 2 + (y - 1) ** 2
# func = lambda x, y: 100 * np.sqrt(np.abs(y - 0.01 * x ** 2)) + 0.01 * np.abs(x + 10)  # Bukin function N.6 min = 0
# func = lambda x, y: - np.cos(x) * np.cos(y) * np.exp(- ((x - np.pi) ** 2 + (y - np.pi) ** 2))  # Easom function  min -1


class GeneticIslands:
    def __init__(self, func, n=40, n_iter=10000, n_islands=5, init_mult=10, exch=2):
        self.func = func
        self.n = n
        self.n_iter = n_iter
        self.population = []
        self.n_islands = n_islands
        self.init_mult = init_mult
        self.generate_random_population()
        self.ex = exch
        self.best_ind = None
        self.best_mean = None
        self.top = 0

    # начальная случайная популяция
    def generate_random_population(self):
        population = np.random.rand(self.n_islands, self.n, 2) * self.init_mult
        self.population = population.tolist()

    def sort_fitness(self, population):
        sorted_population = []
        for i, isl in enumerate(population):
            # print(isl)
            fitness = [(j, self.func(ind[0], ind[1])) for j, ind in enumerate(isl)]
            # print(len(fitness))
            sorted_fitness = sorted(fitness, key=lambda x: x[1])
            # print(sorted_fitness)
            sorted_population.append([isl[num[0]] for num in sorted_fitness])
        return sorted_population

    def selection(self, population):
        for i, island in enumerate(self.sort_fitness(population)):
            self.population[i] = island[:self.n]

    def epoch(self):
        for i in tqdm(range(self.n_iter)):
            if i % 100 == 0 and i != 0:
                self.exchange()
            newPop = copy.deepcopy(self.population)
            if np.random.rand() < 1 / (self.n * self.n_islands):
                self.mutate(newPop)
            for j, island in enumerate(self.population):
                ind = 0
                for _ in range(self.n):
                    first_index = np.random.randint(len(island))
                    a = island[first_index]
                    candidates = [(p, np.sqrt((a[0] - p[0])**2) + (a[1] - p[1])**2) for k, p in enumerate(island) if k != first_index]
                    sorted_cand = sorted(candidates, key=lambda x: x[1])
                    b = sorted_cand[-1][0]
                    c, d = self.crossover(a, b)
                    newPop[j].append(c)
                    newPop[j].append(d)
                    ind += 1
                    if ind >= self.n:
                        break
            self.selection(newPop)
            islands_best = [(island[0], self.func(island[0][0], island[0][1])) for island in self.population]
            islands_best = sorted(islands_best, key=lambda x: x[1])
            best_mean_point = np.mean([ind[0] for ind in islands_best], axis=0)
            new_best = islands_best[0][0]
            if self.best_ind is not None and self.best_mean is not None and new_best == self.best_ind and abs(sum(self.best_mean - best_mean_point)) < 10 ** -5:
                self.top += 1
            else:
                self.best_ind = new_best
                self.best_mean = best_mean_point
                self.top = 0
            if self.top > self.n_islands * 100:
                break
        print(self.best_ind, self.func(self.best_ind[0], self.best_ind[1]))
        return self.best_ind

    @staticmethod
    def crossover(a, b):
        o1, o2 = deap.tools.cxSimulatedBinary(a, b, 2)

        return o1, o2

    @staticmethod
    def sigma():
        m = 20
        s = 0
        for i in range(m):
            if np.random.random() > 1 / m:
                s += 2 ** (-i)
        return s

    def mutate(self, population):
        popul = copy.deepcopy(population)
        sorted_pop = self.sort_fitness(popul)

        for i, island in enumerate(sorted_pop):
            a = lambda: (-1 if np.random.random() < 0.5 else 1) * 0.5 * self.sigma()
            mutated_population = np.array([np.array([i[0] + a(), i[1] + a()]) for i in island[1:]])
            self.population[i] = np.vstack((np.array(island[0]), mutated_population)).tolist()

    def exchange(self):
        pop_copy = copy.deepcopy(self.population)
        for comb in combinations(range(self.n_islands), 2):
            for i in range(self.ex):
                self.population[comb[0]][i] = pop_copy[comb[1]][i]


def process_function(func, x_lim, y_lim, init_mult, n_islands=5, n_iter=10000, n=20):
    g = GeneticIslands(func=func, init_mult=init_mult, n_islands=n_islands, n_iter=n_iter, n=n)
    g.epoch()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(x_lim[0], x_lim[1], 0.1)
    Y = np.arange(y_lim[0], y_lim[1], 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y)
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    for island in g.population:
        x_coords = np.array([ind[0] for ind in island])
        y_coords = np.array([ind[1] for ind in island])
        z_coords = func(x_coords, y_coords)
        ax.scatter(x_coords, y_coords, z_coords)
    plt.show()


if __name__ == '__main__':
    process_function(bukin6, (-15, -5), (-3, 3), -20, 8, 100000, 50)
    # process_function(holder_table, (-10, 10), (-10, 10), 20)
    # process_function(cross_in_tray, (-10, 10), (-10, 10), 20)
