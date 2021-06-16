import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ConfigSpace

from torch.distributions import Categorical

from methods.common import print_stats

BUFFER_SIZE = int(2e4)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 1e-3  # learning rate
UPDATE_EVERY = 4  # how often to update the network


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


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, ops, device, seed=0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.ops = nn.ParameterList([torch.nn.Parameter(torch.zeros(op)) for op in ops])
        self.ops_sampled = None
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

    def set_ops(self):
        dists = [Categorical(logits=op) for op in self.ops]
        self.ops_sampled = torch.cat([di.sample().reshape(1) for di in dists])
        # self.ops_sampled = torch.from_numpy(np.array([0, 0, 1, 1, 2]))

        return self.ops_sampled

    def get_state(self, ops, edges):
        edges = torch.from_numpy(edges)
        state = torch.cat([edges, ops]).unsqueeze(0).float()

        return state.to(self.device)

    def act(self, state, eps=0.):
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
        e = self.experience(state.reshape(1, -1), action, reward, next_state.reshape(1, -1), done)
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

    def __init__(self, edges, b):
        self.bench = b
        self.edges = edges

    def reset(self):
        return np.zeros(self.edges, dtype=np.int8)

    def get_reward(self, states):
        states = [int(state) for state in states.tolist()[0]]
        config = ConfigSpace.Configuration(self.bench.get_configuration_space(), vector=states)
        y, c = self.bench.objective_function_from_config(config, cond_record=6)
        # y, c = self.bench.objective_function_from_matrix(matrix)
        fitness = 1 - float(y)
        return fitness, c

    def step(self, state, action):
        next_state = state.clone()
        next_state[:,action] = 1
        reward, train_time = self.get_reward(next_state)
        done = False
        if train_time == 0:
            done = True

        return next_state, reward, done


def run_rl_dqn(runtime, b, cs):
    #
    VERTICES = 7
    ops = []
    edges = []
    for h in cs.get_hyperparameters():
        if 'op' in h.name:
            ops.append(h.num_choices)
        elif 'edge' in h.name:
            edges.append(h.num_choices)
    #
    env = Environment(edges=len(edges), b=b)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(len(ops) + len(edges), len(edges), ops, device)

    last_time = 0
    eps_start = 1.0
    eps_end = 0.005
    eps_decay = 0.995
    scores = []
    scores_window = collections.deque(maxlen=100)
    eps = eps_start
    while b.get_runtime() < runtime:
        last_time = print_stats(b, last_time)
        #
        edges = env.reset()
        ops_sampled = agent.set_ops()
        state = agent.get_state(ops_sampled, edges)
        score = 0
        cnt = 0
        while cnt < 9:
            # record only when cnt >= 6?
            action = agent.act(state, eps)
            next_state, reward, done = env.step(state, action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            cnt += 1
            # if done:
            #     break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

    return
