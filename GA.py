import numpy as np
import matplotlib.pyplot as plt


def mutate(parents, fitness_function):
    n = int(len(parents))
    scores = fitness_function(parents)
    idx = scores > 0
    scores = scores[idx]
    parents = np.array(parents)[idx]

    children = np.random.choice(parents, size=n, p=scores/scores.sum())
    children = children + np.random.uniform(-0.51, 0.51, size=n)
    return children.tolist()


def genetic_algorithm(parents, fitness_function, max_iter=100):
    history = []
    best_parent, best_fitness = _get_fittest_parent(parents, fitness_function)
    print('generation {}| best fitness {}| current fitness {}| current_parent {}'.format(0, best_fitness, best_fitness,
                                                                                         best_fitness))

    x = np.linspace(start=0, stop=1, num=4)
    plt.plot(x, fitness_function(x))
    plt.scatter(parents, fitness_function(parents), marker='x')

    for i in range(1, max_iter):
        parents = mutate(parents, fitness_function)

        curr_parent, curr_fitness = _get_fittest_parent(parents, fitness_function)

        if curr_fitness < best_fitness:
            best_fitness = curr_fitness
            best_parent = curr_parent

        if i % 10 == 0:
            print('generation {}| best fitness {}| current fitness {}| current parent {}'.format(i, best_fitness,
                                                                                                 curr_fitness,
                                                                                                 curr_parent))
        history.append((i, np.max(fitness_function(parents))))

    plt.scatter(parents, fitness_function(parents))
    plt.scatter(best_parent, fitness_function(best_parent), marker='.', c='b', s=200)
    plt.pause(0.09)
    plt.ioff()

    print('generation {}| best fitness {}| best parent {}'.format(i, best_fitness, best_parent))

    return best_parent, best_fitness, history


def _get_fittest_parent(parents, fitness_function):
    _fitness = fitness_function(parents)
    PFitness = list(zip(parents, _fitness))
    PFitness.sort(key=lambda x: x[1], reverse=True)
    best_parent, best_fitness = PFitness[0]
    return round(best_parent, 4), round(best_fitness, 4)

