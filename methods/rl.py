import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

from methods.common import print_stats

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
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
