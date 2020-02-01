import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



class Fully_Model(nn.Module):
    def __init__(self, state_size, action_size, dinoMap_size, seed, fc_unit=4):
        super(Fully_Model,self).__init__() # calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)

        self.fc = nn.Linear(state_size, fc_unit)
        self.out = nn.Linear(fc_unit, action_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.out(x)
