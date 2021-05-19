import collections
import random
from copy import deepcopy

import ConfigSpace
import numpy as np


# common functions
def random_architecture(cs):
    config = cs.sample_configuration()
    return config


def print_stats(b, last_time, offset=1e5):
    if b.get_runtime() - last_time > offset:
        last_time = b.get_runtime()
        regret_validation, regret_test = b.get_regret()
        print('runtime: %.4f regret_validation: %.4f regret_test: %.4f'
              % (b.get_runtime(), regret_validation, regret_test))

    return last_time


# random search
def random_search(runtime, b, cs):
    last_time = 0
    while b.get_runtime() < runtime:
        last_time = print_stats(b, last_time)
        b.objective_function(random_architecture(cs))


# regularized evolution
class Model(object):

    def __init__(self):
        self.arch = None
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return '{0:b}'.format(self.arch)


def train_and_eval(config, b):
    y, cost = b.objective_function(config)
    # returns negative error (similar to maximizing accuracy)
    # return -y
    return 1 - y, cost


def mutate_arch(parent_arch, cs):
    # pick random parameter
    dim = np.random.randint(len(cs.get_hyperparameters()))
    hyper = cs.get_hyperparameters()[dim]

    if type(hyper) == ConfigSpace.OrdinalHyperparameter:
        choices = list(hyper.sequence)
    else:
        choices = list(hyper.choices)
    # drop current values from potential choices
    choices.remove(parent_arch[hyper.name])

    # flip parameter
    idx = np.random.randint(len(choices))

    child_arch = deepcopy(parent_arch)
    child_arch[hyper.name] = choices[idx]
    return child_arch


def regularized_evolution(b, cs, runtime, population_size=100, sample_size=10):
    population = collections.deque()

    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture(cs)
        model.accuracy, cost = train_and_eval(model.arch, b)
        if cost > 0:
            population.append(model)

    last_time = 0
    while b.get_runtime() < runtime:
        last_time = print_stats(b, last_time)
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch, cs)
        child.accuracy, cost = train_and_eval(child.arch, b)
        population.append(child)

        # Remove the oldest model.
        population.popleft()

    return
