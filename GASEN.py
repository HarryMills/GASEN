# Genetic Algorithm Selection of Ensemble Networks
#
# This script uses a genetic algorithm to selection the optimum
# weighting of networks in an ensemble. The algorithm starts with random
# weights and then starts to minimise the selected value in the
# function ensemble_fitness.

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
from sklearn import linear_model

from loadData import load_data
from ensembleFitness import ensemble_fitness
from weightedEnsemble import weighted_ensemble
import GA

UPPER_BOUND = 1
LOWER_BOUND = 0

# Loading in data
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Predict field
predict = "G3"

# Dropping out the predicted field
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Retrieve network objects from networks folder
models = load_data()

# Create objective function
objective_function = lambda w: ensemble_fitness(w, models, x_test, y_test, 'mse')

# Set Genetic Algorithm parameters
sol_per_pop = 8
num_parents_mating = 4

# Defining population size
pop_size = (sol_per_pop, len(models))
# Creating the initial population
new_population = np.random.uniform(low=0, high=1, size=pop_size)
print(new_population)

num_generations = 100

for generation in range(num_generations):
    print("Generation: ", generation)
    # Measuring the fitness of each chromosome in the population
    fitness = GA.cal_pop_fitness(objective_function, new_population)

    # Selecting the best parents in the population for mating
    parents = GA.select_mating_pool(new_population, fitness, num_parents_mating)

    # Generating next generation using crossover
    offspring_crossover = GA.crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], len(models)))

    # Adding some variations to the offspring using mutation
    offspring_mutation = GA.mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    # The best result in the current iteration
    print("NOTHING")

# Get the best solution after all generations
fitness = GA.cal_pop_fitness(objective_function, new_population)
# Return the index of that solution and corresponding best fitness
best_match_idx = np.where(fitness == np.min(fitness))
print(best_match_idx)
print(fitness)
# Return weights

# Display optimised network ensemble accuracy details
# results = weighted_ensemble(weights, models, x_test)
# pyplot.scatter(results, y_test)
# pyplot.show()
