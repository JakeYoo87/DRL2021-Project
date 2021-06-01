import ConfigSpace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from methods.common import print_stats


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
        # self._numerator = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        # self._denominator = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._numerator = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        self._denominator = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        self._momentum = momentum

    def update(self, value):
        """Update the moving average with a new sample."""
        self._numerator = self._momentum * self._numerator + (1 - self._momentum) * value
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / (self._denominator + 1e-8)


class NaiveReinforce(nn.Module):
    def __init__(self, cat_variables, reward_func):
        super().__init__()
        self._logits = nn.ParameterList([torch.nn.Parameter(torch.zeros(ci)) for ci in cat_variables])
        self._reward_func = reward_func

    def step(self):
        # dists = [Categorical(F.softmax(li)) for li in self._logits]
        dists = [Categorical(logits=li) for li in self._logits]
        while True:
            action = [di.sample() for di in dists]
            # Compute the sample reward. Larger rewards are better.
            reward = self._reward_func.compute_reward(action)
            if reward > 0.001:
                break

        log_prob = sum([dists[i].log_prob(action[i]) for i in range(len(action))])

        return action, log_prob, reward


def run_reinforce(runtime, b, cs):
    # tf.enable_eager_execution()
    # tf.enable_resource_variables()
    nb_reward = Reward(b)
    #
    cat_variables = []
    for h in cs.get_hyperparameters():
        if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
            cat_variables.append(len(h.sequence))
        elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
            cat_variables.append(len(h.choices))

    baseline = ExponentialMovingAverage(momentum=0.9)
    net = NaiveReinforce(cat_variables, nb_reward)
    optimizer = optim.AdamW(net.parameters(), lr=0.001)

    last_time = 0
    while b.get_runtime() < runtime:
        last_time = print_stats(b, last_time)
        optimizer.zero_grad()
        # Compute the gradient of the sample's log-probability w.r.t. the logits.
        action, log_prob, reward = net.step()
        # Compute the log-likelihood the sample.
        # log_prob = (tf.reduce_sum(edge_dist.log_prob(edge_sample)) + tf.reduce_sum(op_dist.log_prob(op_sample)))
        # Update the baseline to reflect the current sample.
        baseline.update(reward)
        # Advantage will be positive if the current sample is better than average.
        advantage = reward - baseline.value()
        # Here comes the REINFORCE magic. We'll update the gradients by differentiating with respect to this value.
        # If advantage > 0, then the update will increase the log-probability,
        # If advantage < 0, then the update will decrease the log-probability.
        loss = advantage * log_prob
        loss.backward()
        optimizer.step()

    return
