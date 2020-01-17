import random
import numpy as np
import copy
#import tensorflow as tf

from collections import namedtuple, deque

from model import Fully_Model

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1)  #replay buffer size
BATCH_SIZE = 1       # minibatch size
GAMMA = 0.8            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 1        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Random_Agent():
    def __init__(self,action_size):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.action_size = action_size

    def step(self,state, action, reward, next_state, done):
        self.t_step +=1

    def act(self, state, done):
        return random.choice(np.arange(self.action_size))

class Sweep_Agent():
    def __init__(self,action_size):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.action_size = action_size
        self.seq = []

        for j in range(4):
            for i in range(7):
                self.seq.append(1)
            self.seq.append(2)
            for i in range(7):
                self.seq.append(3)
            self.seq.append(2)
        self.seq.pop(-1)

    def step(self,state, action, reward, next_state, done):
        self.t_step +=1

    def act(self, state, done):
        if self.seq:
            return self.seq.pop(0)
        else:
            for j in range(4):
                for i in range(7):
                    self.seq.append(1)
                self.seq.append(2)
                for i in range(7):
                    self.seq.append(3)
                self.seq.append(2)
            self.seq.pop(-1)

class Q_Agent():
    def __init__(self,state_size, action_size, dinotype_num, seed):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.action_size = action_size
        #self.Q_table = np.full((8,8,2,2,2,2,4),0.001) #state_size = 8
        self.Q_table = np.full((8, 8, 4), 0.001)
        self.last_action = None
        self.last_observation = None
        self.statespace = []
        self.accumulated_reward = 0
        self.epsilon = 0.05
        self.gamma = 0.8
        self.alpha = 0.9
        #self.decay = 0.99

        self.probMap = np.full((8, 8), 0.0)  # state_size = 8
        #self.visitMap = np.full((8, 8), 0)

    def step(self,state, action, reward, next_state, done, test, map_update):
        self.t_step += 1
        last_state = copy.deepcopy(state)
        curr_state = next_state
        self.accumulated_reward += reward

        # update Q value
        pre_Q = copy.deepcopy(self.get_Q_value(last_state, self.last_action))

        # Q learning
        max_Q = []
        for i in range(self.action_size):
            max_Q.append(self.get_Q_value(curr_state, i))

        #self.alpha = max(self.alpha * self.decay, 0.2)
        new_Q = pre_Q + self.alpha * (reward + self.gamma * max(max_Q) - pre_Q)

        if not test:
            self.Q_table[last_state[0]][last_state[1]][self.last_action] = new_Q
            #self.Q_table[last_state[0]][last_state[1]][last_state[2]][last_state[3]][last_state[4]][last_state[5]][self.last_action] = new_Q
        #     if not self.visitMap[curr_state[0]][curr_state[1]] and reward != -1:
        #         self.visitMap[curr_state[0]][curr_state[1]]  = 1
        #         self.probMap[curr_state[0]][curr_state[1]]  = self.probMap[curr_state[0]][curr_state[1]] + 1


    def get_Q_value(self, state, action):
        # print "battery level",battery
        q_val = self.Q_table[state[0]][state[1]][action]
        #q_val = np.take(self.Q_table, state)
        #q_val = self.Q_table[state[0]][state[1]][state[2]][state[3]][state[4]][state[5]][action]
        return q_val


    def act(self, state):
        curr_state = state
        new_action = None

        if random.random() > self.epsilon:
            max_Q = []
            for i in range(self.action_size):
                max_Q.append(self.get_Q_value(curr_state, i))

            temp = np.argmax(max_Q)
        else:
            temp = random.randrange(self.action_size)

        new_action = temp
        self.last_action = new_action
        self.last_observation = copy.deepcopy(state)
        return new_action

    def reset(self):
        self.t_step = 0
        self.Q_table = np.full((8,8,4),0.001) #state_size = 8
        #self.Q_table = np.full((8, 8, 2, 2, 2, 2, 4), 0.001)  # state_size = 8
        self.last_action = None
        self.last_observation = None
        self.statespace = []
        self.accumulated_reward = 0
        self.alpha = 0.9

class Q_Agent_Neighbour():
    def __init__(self,state_size, action_size, dinotype_num, seed):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.action_size = action_size
        self.Q_table = np.full((8,8,2,2,2,2,4),0.001) #state_size = 8
        self.last_action = None
        self.last_observation = None
        self.statespace = []
        self.accumulated_reward = 0
        self.epsilon = 0.05
        self.gamma = 0.8
        self.alpha = 0.9
        self.decay = 0.99

        self.probMap = np.full((8, 8), 0.0)  # state_size = 8
        #self.visitMap = np.full((8, 8), 0)

    def step(self,state, action, reward, next_state, done, test, map_update):
        self.t_step += 1
        last_state = copy.deepcopy(state)
        curr_state = next_state
        self.accumulated_reward += reward

        # update Q value
        pre_Q = copy.deepcopy(self.get_Q_value(last_state, action))

        # Q learning
        max_Q = []
        for i in range(self.action_size):
            max_Q.append(self.get_Q_value(curr_state, i))

        self.alpha = max(self.alpha * self.decay, 5e-4)
        new_Q = pre_Q + self.alpha * (reward + self.gamma * max(max_Q) - pre_Q)

        if not test:
            #self.Q_table[last_state[0]][last_state[1]][self.last_action] = new_Q
            self.Q_table[last_state[0]][last_state[1]][last_state[2]][last_state[3]][last_state[4]][last_state[5]][self.last_action] = new_Q
        #     if not self.visitMap[curr_state[0]][curr_state[1]] and reward != -1:
        #         self.visitMap[curr_state[0]][curr_state[1]]  = 1
        #         self.probMap[curr_state[0]][curr_state[1]]  = self.probMap[curr_state[0]][curr_state[1]] + 1


    def get_Q_value(self, state, action):
        # print "battery level",battery
        #q_val = self.Q_table[state[0]][state[1]][action]
        #q_val = np.take(self.Q_table, state)
        q_val = self.Q_table[state[0]][state[1]][state[2]][state[3]][state[4]][state[5]][action]
        return q_val


    def act(self, state):
        curr_state = state
        new_action = None

        if random.random() > self.epsilon:
            max_Q = []
            for i in range(self.action_size):
                max_Q.append(self.get_Q_value(curr_state, i))

            temp = np.argmax(max_Q)
        else:
            temp = random.randrange(self.action_size)

        new_action = temp
        self.last_action = new_action
        self.last_observation = copy.deepcopy(state)
        return new_action

    def reset(self):
        self.t_step = 0
        #self.Q_table = np.full((8,8,4),0.001) #state_size = 8
        self.Q_table = np.full((8, 8, 2, 2, 2, 2, 4), 0.001)  # state_size = 8
        self.last_action = None
        self.last_observation = None
        self.statespace = []
        self.accumulated_reward = 0
        self.alpha = 0.9

class Q_Agent_Vector():
    def __init__(self,state_size, action_size, dinotype_num, seed):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.action_size = action_size
        self.Q_table = np.full((8,8,2,2,2,2,2,2,2,2,2,2,2,2,2,4),0.001) #state_size = 8
        self.last_action = None
        self.last_observation = None
        self.statespace = []
        self.accumulated_reward = 0
        self.epsilon = 0.05
        self.gamma = 0.8
        self.alpha = 0.9
        self.decay = 0.99

        self.probMap = np.full((8, 8), 0.0)  # state_size = 8
        #self.visitMap = np.full((8, 8), 0)

    def step(self,state, action, reward, next_state, done, test, map_update):
        self.t_step += 1
        last_state = copy.deepcopy(state)
        curr_state = next_state
        self.accumulated_reward += reward

        # update Q value
        pre_Q = copy.deepcopy(self.get_Q_value(last_state, action))

        # Q learning
        max_Q = []
        for i in range(self.action_size):
            max_Q.append(self.get_Q_value(curr_state, i))

        self.alpha = max(self.alpha * self.decay, 5e-4)
        new_Q = pre_Q + self.alpha * (reward + self.gamma * max(max_Q) - pre_Q)

        if not test:
            #self.Q_table[last_state[0]][last_state[1]][self.last_action] = new_Q
            self.Q_table[last_state[0]][last_state[1]][last_state[2]][last_state[3]][last_state[4]][last_state[5]][last_state[6]][last_state[7]][last_state[8]][last_state[9]][last_state[10]][last_state[11]][last_state[12]][last_state[13]][last_state[14]][self.last_action] = new_Q
        #     if not self.visitMap[curr_state[0]][curr_state[1]] and reward != -1:
        #         self.visitMap[curr_state[0]][curr_state[1]]  = 1
        #         self.probMap[curr_state[0]][curr_state[1]]  = self.probMap[curr_state[0]][curr_state[1]] + 1


    def get_Q_value(self, state, action):
        # print "battery level",battery
        #q_val = self.Q_table[state[0]][state[1]][action]
        #q_val = np.take(self.Q_table, state)
        q_val = self.Q_table[state[0]][state[1]][state[2]][state[3]][state[4]][state[5]][state[6]][state[7]][state[8]][state[9]][state[10]][state[11]][state[12]][state[13]][state[14]][action]
        return q_val


    def act(self, state):
        curr_state = state
        new_action = None

        if random.random() > self.epsilon:
            max_Q = []
            for i in range(self.action_size):
                max_Q.append(self.get_Q_value(curr_state, i))

            temp = np.argmax(max_Q)
        else:
            temp = random.randrange(self.action_size)

        new_action = temp
        self.last_action = new_action
        self.last_observation = copy.deepcopy(state)
        return new_action

    def reset(self):
        self.t_step = 0
        #self.Q_table = np.full((8,8,4),0.001) #state_size = 8
        self.Q_table = np.full((8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4), 0.001)
        self.last_action = None
        self.last_observation = None
        self.statespace = []
        self.accumulated_reward = 0
        self.alpha = 0.9

class Q_Agent_single_prob():
    def __init__(self,state_size, action_size, dinotype_num, seed):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.action_size = action_size
        #self.Q_table = np.full((8,8,2,2,2,2,4),0.001) #state_size = 8
        self.Q_table = np.full((8, 8, 2, 4), 0.001)
        self.last_action = None
        self.last_observation = None
        self.statespace = []
        self.accumulated_reward = 0
        self.epsilon = 0.05
        self.gamma = 0.8
        self.alpha = 0.9
        #self.decay = 0.99

        self.probMap = np.full((8, 8), 0)  # state_size = 8
        self.visitMap = np.full((8, 8), 0)

    def step(self,state, action, reward, next_state, done, test, map_update):
        self.t_step += 1
        last_state = copy.deepcopy(state)
        curr_state = next_state
        self.accumulated_reward += reward

        # update Q value
        pre_Q = copy.deepcopy(self.get_Q_value(last_state, self.last_action))

        # Q learning
        max_Q = []
        for i in range(self.action_size):
            max_Q.append(self.get_Q_value(curr_state, i))

        #self.alpha = max(self.alpha * self.decay, 0.2)
        new_Q = pre_Q + self.alpha * (reward + self.gamma * max(max_Q) - pre_Q)

        prob_bool = self.probMap[last_state[0]][last_state[1]]

        if not test:
            self.Q_table[last_state[0]][last_state[1]][prob_bool][self.last_action] = new_Q
            #self.Q_table[last_state[0]][last_state[1]][last_state[2]][last_state[3]][last_state[4]][last_state[5]][self.last_action] = new_Q
            if not self.visitMap[curr_state[0]][curr_state[1]] and reward != -1 and not map_update:
                self.visitMap[curr_state[0]][curr_state[1]]  = 1
                self.probMap[curr_state[0]][curr_state[1]]  = 1


    def get_Q_value(self, state, action):
        # print "battery level",battery
        prob_bool = self.probMap[state[0]][state[1]]
        q_val = self.Q_table[state[0]][state[1]][prob_bool][action]
        #q_val = np.take(self.Q_table, state)
        #q_val = self.Q_table[state[0]][state[1]][state[2]][state[3]][state[4]][state[5]][action]
        return q_val


    def act(self, state):
        curr_state = state
        new_action = None

        if random.random() > self.epsilon:
            max_Q = []
            for i in range(self.action_size):
                max_Q.append(self.get_Q_value(curr_state, i))

            temp = np.argmax(max_Q)
        else:
            temp = random.randrange(self.action_size)

        new_action = temp
        self.last_action = new_action
        self.last_observation = copy.deepcopy(state)
        return new_action

    def reset(self):
        self.t_step = 0
        self.Q_table = np.full((8,8,2,4),0.001) #state_size = 8
        #self.Q_table = np.full((8, 8, 2, 2, 2, 2, 4), 0.001)  # state_size = 8
        self.last_action = None
        self.last_observation = None
        self.statespace = []
        self.accumulated_reward = 0
        self.alpha = 0.9

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
        #self.policy_net = Simple_Network(state_size, action_size, dinotype_num,seed).to(device)
        #self.target_net = Simple_Network(state_size, action_size, dinotype_num,seed).to(device)

        self.policy_net = Fully_Model(state_size, action_size, dinotype_num, seed).to(device)
        self.target_net = Fully_Model(state_size, action_size, dinotype_num, seed).to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_step, done,test, map_update):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)
        experience = self.memory.sample()
        self.learn(experience, GAMMA)


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
