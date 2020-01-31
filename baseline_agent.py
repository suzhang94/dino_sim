import random
import numpy as np
import copy
#import tensorflow as tf

from collections import namedtuple, deque

from model import Fully_Model

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e4)  #replay buffer size
BATCH_SIZE = 16      # minibatch size
GAMMA = 0.9            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Fully_Agent_Neighbour():
    def __init__(self, state_size, action_size, dinotype_num,seed):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        #self.probMap = torch.zeros(8, 8, dtype=torch.float)
        self.probMap = np.full((8, 8), 0)

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q- Network
        self.policy_net = Fully_Model(state_size, action_size, dinotype_num, seed).to(device)
        self.target_net = Fully_Model(state_size, action_size, dinotype_num, seed).to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_step, done,test, map_update):
        # Save experience in replay memory

        #loc = np.asarray(state)
        # state = torch.cat((loc_tensor, self.probMap[loc[0]][loc[1]]), 1)
        #prob_vec = np.reshape(self.probMap, 64)

        #state_vector = np.concatenate((loc, prob_vec), axis=None)
        #state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).to(device)

        #next_state_vector = np.concatenate((np.asarray(next_step), prob_vec), axis=None)
        #next_state_tensor = torch.from_numpy(next_state_vector).float().unsqueeze(0).to(device)


        self.memory.add(state, action, reward, next_step, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

        # if not test:
        #     if self.visitMap[state[0]][state[1]] == 0:
        #         self.visitMap[state[0]][state[1]] = 1
        #         if reward != -1:
        #             self.probMap[state[0]][state[1]] = 1
        #         else:
        #             self.probMap[state[0]][state[1]] = 0
        #     else:
        #         self.probMap[state[0]][state[1]] = 0


    def act(self, state, eps):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        loc = np.asarray(state)
        #state = torch.cat((loc_tensor, self.probMap[loc[0]][loc[1]]), 1)

        #prob_vec = np.reshape(self.probMap, 64)

        #state_vector =  np.concatenate((loc, prob_vec), axis=None)
        state_tensor = torch.from_numpy(loc).float().unsqueeze(0).to(device)

        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net.forward(state_tensor)
        self.policy_net.train()

        #Epsilon -greedy action selction
        if random.random() > eps:
            a = np.argmax(action_values.cpu().data.numpy())
        else:
            a = random.choice(np.arange(self.action_size))
        return a

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        self.policy_net.train()
        self.target_net.eval()
        # shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.policy_net, self.target_net, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    pass

class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                                 "action",
                                                                 "reward",
                                                                 "next_state",
                                                                 "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def para_setting(paras):
    global BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY
    BUFFER_SIZE = paras[0]
    BATCH_SIZE = paras[1]
    GAMMA = paras[2]
    TAU = paras[3]
    LR = paras[4]
    UPDATE_EVERY = paras[5]
    pass

def para_print():
    global BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY
    print(BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY)
    pass