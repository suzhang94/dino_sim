from collections import deque
import torch
import numpy as np
import matplotlib.pyplot as plt
import util

import env
#import agent
import baseline_agent as agent

#agent = agent.Fully_Agent(state_size=4, action_size=4, dinotype_num=8, seed=0)
#agent = agent.Q_Agent_Neighbour(state_size=6, action_size=4, dinotype_num=8, seed=0)
#agent = agent.Q_Agent_Vector(state_size=15, action_size=4, dinotype_num=8, seed=0)
#agent = agent.Q_Agent_Vector(state_size=15, action_size=4, dinotype_num=8, seed=0)
#agent = agent.Q_Agent_single_prob(state_size=5, action_size=4, dinotype_num=8, seed=0)
agent = agent.Fully_Agent_Neighbour(state_size=4, action_size=4, dinotype_num=8, seed=0)

env = env.Environment(size=8)

map_vec = env.init_map_vec()

def dqn(n_runs = 5, n_episodes=5000, max_t=300, eps_start=1.0, eps_end = 0.05,
       eps_decay=0.996):
    steps = np.zeros(n_episodes)
    acc_rewards = []
    scores = []
    eps = eps_start

    # map_vec = env.init_map_vec()
    # for num in map_vec:
    #     loc = util.num_to_loc(num,8)
    #     agent.probMap[loc[0]][loc[1]] = 1

    #print(agent.probMap)

    for i_run in range(0, n_runs):
        # train
        print("run: ",i_run)
        # provide the learned map
        #agent.reset()
        for i_episode in range(0, n_episodes):
            state = env.reset()
            #score = 0
            for t in range(max_t):
                success = False
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.step(state, action, reward, next_state, done, False, True) # not update the map
                state = next_state
                eps = max(eps * eps_decay, eps_end)
                #score += reward
                if done:
                    #print(env.map)
                    #print("t",t,"score",score)
                    steps[i_episode] = steps[i_episode] + t
                    success = True
                    #print(t)
                    break
            if not success:
                steps[i_episode] = steps[i_episode] + max_t
                #print(t)
            #agent.visitMap = np.full((8, 8), 0)
        #agent.reset()

    return scores,steps, agent.probMap

scores,steps, probMap = dqn()

'''
probMap = np.divide(probMap,500)
print(probMap)
plt.imshow(probMap, cmap='GnBu', interpolation='bilinear')
plt.show()

plt.imshow(probMap, cmap='GnBu', interpolation='nearest')
plt.show()
'''

print(len(steps))

steps = steps/5

print("average steps",sum(steps)/len(steps))

print("probMap",agent.probMap)

fig = plt.figure()
ax = fig.add_subplot(111)

avg_steps = util.running_mean(steps,200)
plt.plot(np.arange(len(steps)-200+1), avg_steps)
plt.ylabel('Steps')
plt.xlabel('# of Episodes')
plt.show()
