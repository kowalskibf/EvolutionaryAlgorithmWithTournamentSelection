import cec2017
import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from cec2017.functions import f2, f13

class Result:
    def __init__(self, function, sigma, mu, iter, individual, fitness):
        self.function = function
        self.sigma = sigma
        self.mu = mu
        self.iter = iter
        self.individual = individual
        self.fitness = fitness

    def __str__(self):
        return f'Function: {self.function}, SIGMA: {self.sigma}, MU: {self.mu}, Iter: {self.iter}, Fitness: {self.fitness}\nIndividual: {self.individual}\n'

    def strOnlyFitness(self):
        return str(self.fitness) + '\n'

def booth(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

def evolutionary_algorithm(function, population, tournament_size, MU, SIGMA, t_max):
    for i in range(t_max):
        fitness = np.array([function(individual) for individual in population])
        winners = []
        for _ in range(MU):
            participants = np.random.choice(MU, tournament_size, replace=False)
            participants_fitness = fitness[participants]
            winner_index = participants[np.argmin(participants_fitness)]
            winners.append(population[winner_index])
        population = np.array(winners) + SIGMA * np.random.randn(MU, dimensions)
    return [population[np.argmin(fitness)], np.min(fitness)]

UPPER_BOUND = 100
BUDGET = 10000
MU = 100
SIGMA = 3
tournament_size = 3
dimensions = 10
t_max = BUDGET // MU

functions = [f2, f13]
SIGMAs = [10e-2, 3, 10, 10e3]

results = []

for fi in range(len(functions)):
    for SIGMA in SIGMAs:
        print(SIGMA)
        for _MU in range(6, 10):
            MU = int(2**_MU)
            print(MU)
            t_max = BUDGET // MU
            res_individuals = []
            res_fitness = []
            for _ in range(25):
                population = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=(MU, dimensions))
                best_individual, best_fitness = evolutionary_algorithm(functions[fi], population, tournament_size, MU, SIGMA, t_max)
                function = 'f2'
                if fi:
                    function = 'f13'
                result = Result(function, SIGMA, MU, _, best_individual, best_fitness)
                results.append(result)
                print(result)
                with open(f'results_function_{function}_sigma_{SIGMA}_MU_{MU}.txt', 'a') as file:
                    file.write(str(result))
                with open(f'results_only_numbers_function_{function}_sigma_{SIGMA}_MU_{MU}.txt', 'a') as file:
                    file.write(result.strOnlyFitness())
                res_individuals.append(result.individual)
                res_fitness.append(result.fitness)
            res_best_fitness = min(res_fitness)
            res_best_fitness_idx = res_fitness.index(res_best_fitness)
            res_worst_fitness = max(res_fitness)
            res_worst_fitness_idx = res_fitness.index(res_worst_fitness)
            res_best_individual = res_individuals[res_best_fitness_idx]
            res_worst_individual = res_individuals[res_worst_fitness_idx]
            res_stddev = np.std(res_fitness)
            res_avg = np.average(res_fitness)
            _res = f'Function: {function}, SIGMA: {SIGMA}, MU: {MU}, Best_ind: {res_best_individual}, Best_fit: {res_best_fitness}, Worst_ind: {res_worst_individual}, Worst_fit: {res_worst_fitness}, Avg: {res_avg}, Std_dev: {res_stddev}\n\n'
            with open(f'results_each.txt', 'a') as file:
                file.write(_res)


for result in results:
    print(result)


    
