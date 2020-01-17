import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Network(nn.Module):
    def __init__(self, state_size, action_size, dinoMap_size,seed, fc_unit = 50,
                 conv1_unit = 16, fc1_unit = 16):
        """
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_unit (int): Number of nodes in first hidden layer
            fc1_unit (int): Number of nodes in second hidden layer
            fc1_unit (int): Number of nodes in second hidden layer
        """
        super(Network,self).__init__() # calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        #shared fully connected layer
        self.fc = nn.Linear(4,fc_unit)

        # policy
        #self.conv1 = nn.Conv2d(fc_unit,conv1_unit,3) # use conv 2d? parameters to be tuned later
        self.fc2 = nn.Linear(50, 32)
        self.head_policy = nn.Linear(32, 4)

        # probability map
        self.fc1 = nn.Linear(fc_unit,16)
        self.head_probMap = nn.Linear(16, 2) # or directly use the linear layer as next input?

    def forward_policy(self, x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        return self.head_policy(x)

    def forward_probMap(self, x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        return self.head_probMap(x)


    def forward(self, x):
        # x = state
        state_0 = x.clone()
        state_1 = x.clone()
        p =  self.forward_policy(state_0)
        self.forward_probMap(state_1)
        # need return or not
        return p
    pass

class Simple_Network(nn.Module):
    def __init__(self, state_size, action_size, dinoMap_size,seed, fc_unit = 16,
                 conv1_unit = 32, fc1_unit = 16):
        """
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_unit (int): Number of nodes in first hidden layer
            fc1_unit (int): Number of nodes in second hidden layer
            fc1_unit (int): Number of nodes in second hidden layer
        """
        super(Simple_Network,self).__init__() # calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        #shared fully connected layer
        self.fc = nn.Linear(2,fc_unit)
        # policy
        #self.conv1 = nn.Conv2d(2, conv1_unit, kernel_size=4, stride=2)
        self.head_policy = nn.Linear(fc_unit, action_size)

    def forward(self, x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc(x))
        return self.head_policy(x)


class Fully_Model(nn.Module):
    def __init__(self, state_size, action_size, dinoMap_size,seed, fc_unit = 4):
        """
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_unit (int): Number of nodes in first hidden layer
            fc1_unit (int): Number of nodes in second hidden layer
            fc1_unit (int): Number of nodes in second hidden layer
        """
        super(Fully_Model,self).__init__() # calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        #shared fully connected layer
        self.fc = nn.Linear(6,4)
        # policy
        #self.conv1 = nn.Conv2d(2, conv1_unit, kernel_size=4, stride=2)
        #self.head_policy = nn.Linear(fc_unit, action_size)

    def forward(self, x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        #x = F.relu(self.fc(x))
        return self.fc(x)