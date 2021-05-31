import collections
import random
from copy import deepcopy

import ConfigSpace
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical


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


# BOHB
from hpbandster.core.worker import Worker
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
import logging


# logging.basicConfig(level=logging.ERROR)


class MyWorker(Worker):

    def __init__(self, b, run_id, id, nameserver, nameserver_port):
        super(MyWorker, self).__init__(run_id=run_id, id=id,
                                       nameserver=nameserver, nameserver_port=nameserver_port)
        self.b = b

    def compute(self, config, budget, **kwargs):
        y, cost = self.b.objective_function(config, budget=108)
        return ({
            'loss': float(y),
            'info': float(cost)})


def run_bohb(runtime, b, cs):
    min_budget = 4
    max_budget = 108
    hb_run_id = '0'
    NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
    ns_host, ns_port = NS.start()
    num_workers = 1
    workers = []

    for i in range(num_workers):
        w = MyWorker(b=b, run_id=hb_run_id, id=i,
                     nameserver=ns_host, nameserver_port=ns_port)
        w.run(background=True)
        workers.append(w)

    bohb = BOHB(configspace=cs, run_id=hb_run_id,
                min_budget=min_budget, max_budget=max_budget,
                nameserver=ns_host, nameserver_port=ns_port,
                ping_interval=10, min_bandwidth=0.3)

    n_iters = 100
    results = bohb.run(n_iters, min_n_workers=num_workers)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()


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


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, device, seed=0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, device, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().view(-1).unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = collections.deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = collections.namedtuple("Experience",
                                                 field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state.reshape(1,-1), action, reward, next_state.reshape(1,-1), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Environment(object):

    def __init__(self, vertice, b):
        self.vertice = vertice
        # self.nb_reward = Reward(b)
        self.bench = b
        self.matrix = np.zeros([self.vertice, self.vertice], dtype=np.int8)

    def reset(self):
        self.matrix = np.zeros([self.vertice, self.vertice], dtype=np.int8)
        self.matrix[0][2] = 1  # input -> 3x3 conv
        self.matrix[2][6] = 1  # 3x3 conv -> output

        return self.matrix

    def get_reward(self, matrix):
        # config = ConfigSpace.Configuration(self.bench.get_configuration_space())
        y, c = self.bench.objective_function_from_matrix(matrix)
        fitness = 1 - float(y)
        return fitness, c

    def step(self, action):
        idx = np.triu_indices(self.vertice, k=1)
        row = idx[0][action]
        col = idx[1][action]
        self.matrix[row][col] = 1
        reward, train_time = self.get_reward(self.matrix)
        done = False
        if train_time == 0:
            done = True

        return self.matrix, reward, done


def run_rl(runtime, b, cs):
    #
    # baseline = ExponentialMovingAverage(momentum=0.9)
    env = Environment(vertice=7, b=b)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(49, 21, device)

    last_time = 0
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    scores = []
    scores_window = collections.deque(maxlen=100)
    eps = eps_start
    while b.get_runtime() < runtime:
        last_time = print_stats(b, last_time)
        #
        matrix = env.reset()
        score = 0
        while np.sum(matrix) <= 9:
            action = agent.act(matrix, eps)
            next_matrix, reward, done = env.step(action)
            agent.step(matrix, action, reward, next_matrix, done)
            matrix = next_matrix
            score += reward
            # if done:
            #     break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

    return
