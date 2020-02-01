import numpy as np
import random
import copy
from collections import namedtuple, deque

# from model import Network

import torch
import torch.nn.functional as F
import torch.optim as optim

import util

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

def generate_map2(size = 8):
    dino_map = np.chararray((size,size),unicode=True)
    dino_map[:] = '*'

    k1 = (3,3)
    k2 = (6,6)
    #while (util.distance(k1,k2)<4):
    #    k1 = (random.randrange(2, size-1), random.randrange(2, size-1))
    #    k2 = (random.randrange(1, size-1), random.randrange(1, size))

    dino_map[k1[0]-1][k1[1]-1] = 'C'
    dino_map[k1[0]-1][k1[1]] = 'C'
    dino_map[k1[0]-1][k1[1]+1] = 'C'
    dino_map[k1[0]][k1[1]-1] = 'C'
    dino_map[k1[0]][k1[1]] = 'C'
    dino_map[k1[0]][k1[1]+1] = 'C'
    dino_map[k1[0]+1][k1[1]-1] = 'C'
    dino_map[k1[0]+1][k1[1]] = 'C'
    dino_map[k1[0]+1][k1[1]+1] = 'C'


    dino_map[k2[0]-1][k2[1]-1] = 'R'
    dino_map[k2[0]-1][k2[1]] = 'R'
    dino_map[k2[0]][k2[1]-1] = 'R'
    dino_map[k2[0]][k2[1]] = 'R'

    return dino_map



def generate_map(size = 8):
    dino_map = np.chararray((size,size),unicode=True)
    dino_map[:] = '*'

    k1 = (2,5)
    k2 = (1,1)
    #while (util.distance(k1,k2)<4):
    #    k1 = (random.randrange(2, size-1), random.randrange(2, size-1))
    #    k2 = (random.randrange(1, size-1), random.randrange(1, size))

    dino_map[k1[0]-1][k1[1]-1] = 'C'
    dino_map[k1[0]-1][k1[1]] = 'C'
    dino_map[k1[0]-1][k1[1]+1] = 'C'
    dino_map[k1[0]][k1[1]-1] = 'C'
    dino_map[k1[0]][k1[1]] = 'C'
    dino_map[k1[0]][k1[1]+1] = 'C'
    dino_map[k1[0]+1][k1[1]-1] = 'C'
    dino_map[k1[0]+1][k1[1]] = 'C'
    dino_map[k1[0]+1][k1[1]+1] = 'C'


    dino_map[k2[0]-1][k2[1]-1] = 'R'
    dino_map[k2[0]-1][k2[1]] = 'R'
    dino_map[k2[0]][k2[1]-1] = 'R'
    dino_map[k2[0]][k2[1]] = 'R'

    return dino_map

class Environment():

    def __init__(self,size):
        """
        Every environment should be derived from gym.Env and at least contain the variables observation_space and action_space
        specifying the type of possible observations and actions using spaces.Box or spaces.Discrete.
        Example:
        """
        self.last_observation = [0,0]
        self.observation = [0,0]
        self.reward = None
        self.size = size
        self.origin_map = generate_map(size)
        self.origin_map2 = generate_map2(size)
        self.map = copy.deepcopy(self.origin_map)
        self.episode_end = False
        self.map_vec = None
        self.render_state = None

    def step(self, action):
        """
        This method is the primary interface between environment and agent.
        Paramters:
            action: int
                    the index of the respective action (if action space is discrete)
        Returns:
            output: (array, float, bool)
                    information provided by the environment about its current state:
                    (observation, reward, done)
        """
        self.reward = 0

        # take action
        if action == 0:
            new_observation = [self.last_observation[0]-1, self.last_observation[1]]
        elif action == 1:
            new_observation = [self.last_observation[0], self.last_observation[1]+1]
        elif action == 2:
            new_observation = [self.last_observation[0]+1, self.last_observation[1]]
        elif action == 3:
            new_observation = [self.last_observation[0], self.last_observation[1]-1]

        # check boundary
        self.observation = util.bound_check(new_observation, self.size)

        if self.map[self.observation[0]][self.observation[1]] == 'C':
            self.reward = 1
            self.map[self.observation[0]][self.observation[1]] = '*'
            #print(self.map)
        elif self.map[self.observation[0]][self.observation[1]] == 'R':
            self.map[self.observation[0]][self.observation[1]] = '*'
            self.reward = 10
            #print(self.map)
        else:
            self.reward = -1

        self.last_observation = self.observation
        if (self.map == '*').all():
            self.episode_end = True

        # only return (x,y)
        #return self.observation, self.reward, self.episode_end

        #return 15-d map vector as state, global view
        # map_vec = []
        #
        # for i in range(len(self.map_vec)):
        #     loc_check = util.num_to_loc(self.map_vec[i], self.size)
        #     if self.map[loc_check[0]][loc_check[1]] != '*':
        #         map_vec.append(1)
        #     else:
        #         map_vec.append(0)
        #
        # map_vec.insert(0, self.observation[1])
        # map_vec.insert(0, self.observation[0])
        #
        # return map_vec, self.reward, self.episode_end

        # return 6-d neighbour vector as state, local view
        neighbour_vec = self.get_neighbour(self.observation)

        neighbour_vec.insert(0, self.observation[1])
        neighbour_vec.insert(0, self.observation[0])

        for i in range(len(neighbour_vec)):
            if neighbour_vec[i] == '*':
                neighbour_vec[i]= 0
            elif neighbour_vec[i] == 'R' :
                neighbour_vec[i]= 1
            elif neighbour_vec[i] == 'C':
                neighbour_vec[i] = 1

        #print(neighbour_vec)
        self.render_state = neighbour_vec

        return neighbour_vec, self.reward, self.episode_end



    def reset(self):
        """
        This method resets the environment to its initial values.
        Returns:
            observation:    array
                            the initial state of the environment
        """
        self.episode_end = False
        #self.last_observation = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        #self.observation = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        #self.last_observation = [0, 0, 0, 0, 0, 0, 0]
        #self.observation = [0, 0, 0, 0, 0, 0]
        self.observation = self.rand_init_loc()
        self.last_observation = self.observation
        self.render_state = self.observation

        #self.last_observation = [0, 0]
        #self.observation = [0, 0]

        self.reward = None
        self.map = copy.deepcopy(self.origin_map)
        self.map_vec = self.init_map_vec()
        #print(self.map)
        return self.observation

    def reset2(self):
        """
        This method resets the environment to its initial values.
        Returns:
            observation:    array
                            the initial state of the environment
        """
        self.episode_end = False
        self.last_observation = [0,0]
        self.observation = [0,0]
        self.reward = None
        self.map = copy.deepcopy(self.origin_map2)
        #print(self.map)
        return self.observation

    def render(self, mode='human', close=False):
        """
        This methods provides the option to render the environment's behavior to a window
        which should be readable to the human eye if mode is set to 'human'.
        """
        print(self.map)
        print(self.render_state)

    def get_neighbour(self, loc):
        neighbour_vec = []
        neighbour_loc = [util.bound_check([loc[0], loc[1] - 1], 8),
                         util.bound_check([loc[0], loc[1] + 1], 8),
                         util.bound_check([loc[0] - 1, loc[1]], 8),
                         util.bound_check([loc[0] + 1, loc[1]], 8)]
        for location in neighbour_loc:
            neighbour_vec.append(self.map[location[0]][location[1]])

        return neighbour_vec

    def rand_init_loc(self):
        loc = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
        neighbour_vec = self.get_neighbour(loc)
        neighbour_vec.insert(0, loc[0])
        neighbour_vec.insert(0, loc[1])
        for i in range(len(neighbour_vec)):
            if neighbour_vec[i] == '*':
                neighbour_vec[i]= 0
            elif neighbour_vec[i] == 'R' :
                neighbour_vec[i]= 1
            elif neighbour_vec[i] == 'C':
                neighbour_vec[i] = 1
        return neighbour_vec

    def init_map_vec(self):
        map_vec = []
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i][j] != '*':
                    map_vec.append(util.loc_to_num([i,j], self.size))
        return map_vec
