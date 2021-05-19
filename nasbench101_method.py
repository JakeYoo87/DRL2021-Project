import collections
import random
from copy import deepcopy

import ConfigSpace
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp


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
    tf.enable_eager_execution()
    tf.enable_resource_variables()


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


# reinforcement learning
class Reward(object):
    """Computes the fitness of a sampled model by querying NASBench."""

    def __init__(self, bench):
        self.bench = bench

    def compute_reward(self, sample):
        config = ConfigSpace.Configuration(self.bench.get_configuration_space(), vector=sample)
        y, c = self.bench.objective_function(config)
        fitness = 1 - float(y)
        return fitness


class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""

    def __init__(self, momentum):
        self._numerator = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._denominator = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._momentum = momentum

    def update(self, value):
        """Update the moving average with a new sample."""
        self._numerator.assign(self._momentum * self._numerator + (1 - self._momentum) * value)
        self._denominator.assign(self._momentum * self._denominator + (1 - self._momentum))

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator


class REINFORCEOptimizer(object):
    """Class that optimizes a set of categorical variables using REINFORCE."""

    def __init__(self, reward, cat_variables, momentum):
        # self._num_vertices = reward.num_vertices
        # self._num_operations = len(reward.available_ops)
        # self._num_edges = (self._num_vertices * (self._num_vertices - 1)) // 2
        #
        # self._edge_logits = tf.Variable(tf.zeros([self._num_edges, 2]))
        # self._op_logits = tf.Variable(tf.zeros([self._num_vertices - 2,
        #                                         self._num_operations]))
        self._num_variables = len(cat_variables)
        self._logits = [tf.Variable(tf.zeros([1, ci])) for ci in cat_variables]
        self._baseline = ExponentialMovingAverage(momentum=momentum)
        self._reward = reward
        self._last_reward = 0.0
        self._test_acc = 0.0

    def step(self):
        """Helper function for a single step of the REINFORCE algorithm."""
        # Obtain a single sample from the current distribution.
        # edge_dist = tfp.distributions.Categorical(logits=self._edge_logits)
        # op_dist = tfp.distributions.Categorical(logits=self._op_logits)
        dists = [tfp.distributions.Categorical(logits=li) for li in self._logits]
        attempts = 0
        while True:
            sample = [di.sample() for di in dists]

            # Compute the sample reward. Larger rewards are better.
            reward = self._reward.compute_reward(sample)
            attempts += 1
            if reward > 0.001:
                # print('num attempts: {}, reward: {}'.format(str(attempts), reward))
                break

        self._last_reward = reward

        # Compute the log-likelihood the sample.
        log_prob = tf.reduce_sum([dists[i].log_prob(sample[i]) for i in range(len(sample))])
        # log_prob = (tf.reduce_sum(edge_dist.log_prob(edge_sample)) +
        #             tf.reduce_sum(op_dist.log_prob(op_sample)))

        # Update the baseline to reflect the current sample.
        self._baseline.update(reward)

        # Compute the advantage. This will be positive if the current sample is
        # better than average, and will be negative otherwise.
        advantage = reward - self._baseline.value()

        # Here comes the REINFORCE magic. We'll update the gradients by
        # differentiating with respect to this value. In practice, if advantage > 0
        # then the update will increase the log-probability, and if advantage < 0
        # then the update will decrease the log-probability.
        objective = tf.stop_gradient(advantage) * log_prob

        return objective

    def trainable_variables(self):
        # Return a list of trainable variables to update with gradient descent.
        # return [self._edge_logits, self._op_logits]
        return self._logits

    def baseline(self):
        """Return an exponential moving average of recent reward values."""
        return self._baseline.value()

    def last_reward(self):
        """Returns the last reward earned."""
        return self._last_reward

    def test_acc(self):
        """Returns the last test accuracy computed."""
        return self._test_acc

    def probabilities(self):
        """Return a set of probabilities for each learned variable."""
        # return [tf.nn.softmax(self._edge_logits),
        #        tf.nn.softmax(self._op_logits)]
        # return tf.nn.softmax(self._op_logits)  # More interesting to look at ops
        return [tf.nn.softmax(li).numpy() for li in self._logits]


def run_reinforce(optimizer, learning_rate, max_time, bench, num_steps, log_every_n_steps=1000):
    """Run multiple steps of REINFORCE to optimize a fixed reward function."""
    trainable_variables = optimizer.trainable_variables()
    trace = []
    # run = [[0.0, 0.0, 0.0]]

    # step = 0
    for step in range(num_steps):
        # step += 1
        # Compute the gradient of the sample's log-probability w.r.t. the logits.
        with tf.GradientTape() as tape:
            objective = optimizer.step()

        # Update the logits using gradient ascent.
        gradients = tape.gradient(objective, trainable_variables)
        for grad, var in zip(gradients, trainable_variables):
            var.assign_add(learning_rate * grad)

        trace.append(optimizer.probabilities())
        # run.append([nasbench.training_time_spent,
        #             optimizer.last_reward(),  # validation acc
        #             optimizer.test_acc()])  # test acc (avg)
        if step % log_every_n_steps == 0:
            print('step = {:d}, baseline reward = {:.5f}'.format(
                step, optimizer.baseline().numpy()))
        # if nasbench.training_time_spent > max_time:
        #     break

    return trace


def run_rl(runtime, b, cs):
    # Eager mode used for RL baseline
    tf.enable_eager_execution()
    tf.enable_resource_variables()
    nb_reward = Reward(b)
    #
    cat_variables = []
    for h in cs.get_hyperparameters():
        if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
            cat_variables.append(len(h.sequence))
        elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
            cat_variables.append(len(h.choices))

    optimizer = REINFORCEOptimizer(reward=nb_reward, cat_variables=cat_variables, momentum=0.9)
    trace = run_reinforce(
        optimizer=optimizer,
        learning_rate=1e-2,
        max_time=5e6,
        bench=b,
        num_steps=100,
        log_every_n_steps=100)

    return trace
