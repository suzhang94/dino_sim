from collections import deque
import torch
import numpy as np
import matplotlib.pyplot as plt
import util

import env
#import agent
import baseline_agent

#agent = agent.Fully_Agent(state_size=4, action_size=4, dinotype_num=8, seed=0)
#agent = agent.Q_Agent_Neighbour(state_size=6, action_size=4, dinotype_num=8, seed=0)
#agent = agent.Q_Agent_Vector(state_size=15, action_size=4, dinotype_num=8, seed=0)
#agent = agent.Q_Agent_Vector(state_size=15, action_size=4, dinotype_num=8, seed=0)
#agent = agent.Q_Agent_single_prob(state_size=5, action_size=4, dinotype_num=8, seed=0)
agent = baseline_agent.Fully_Agent_Neighbour(state_size=4, action_size=4, dinotype_num=8, seed=0)

env = env.Environment(size=8)

map_vec = env.init_map_vec()

def dqn(n_runs, n_episodes, max_t=300, eps_start=0.05, eps_end = 1e-4, eps_decay=0.996):
    steps = np.zeros(n_episodes)
    acc_rewards = []
    scores = []
    eps = eps_start

    map_vec = env.init_map_vec()
    probMap = np.full((8, 8), 0)
    for num in map_vec:
        loc = util.num_to_loc(num,8)
        probMap[loc[0]][loc[1]] = 1

    print(agent.probMap)

    for i_run in range(0, n_runs):
        # train
        print("run: ",i_run)
        # provide the learned map
        #agent.reset()
        for i_episode in range(0, n_episodes):
            if i_episode%500 == 0:
                print(i_episode)
            state = env.reset()
            #score = 0
            #agent.probMap = probMap
            #agent.visitMap = np.full((8, 8), 0)
            for t in range(max_t):
                success = False
                action = agent.act(state, eps)
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

        #agent.reset()

    return scores,steps, agent.probMap



'''
probMap = np.divide(probMap,500)
print(probMap)
plt.imshow(probMap, cmap='GnBu', interpolation='bilinear')
plt.show()

plt.imshow(probMap, cmap='GnBu', interpolation='nearest')
plt.show()
'''
#n_runs = 3, n_episodes = 5000, max_t = 300, eps_start = 0.05, eps_end = 1e-4, eps_decay = 0.996

def bash_func(run_settings, paras):
    n_runs = run_settings[0]
    n_episodes = run_settings[1]

    baseline_agent.para_print()
    baseline_agent.para_setting(paras)
    baseline_agent.para_print()

    scores, steps, probMap = dqn(run_settings[0],run_settings[1],run_settings[2],run_settings[3],run_settings[4],run_settings[5])
    #print(len(steps))

    steps = steps / n_runs
    print("average steps", sum(steps) / len(steps))

    # print("probMap",agent.probMap)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    avg_steps = util.running_mean(steps, 200)
    plt.plot(np.arange(len(steps) - 200 + 1), avg_steps)
    plt.ylabel('Steps')
    plt.xlabel('# of Episodes')
    #plt.show()
    path = str(run_settings)+str(paras)
    plt.savefig(path+'.png')


run_settings = [10, 3000, 300, 0.005, 0.0001, 0.996]
#BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY
paras = [10000, 16, 0.95, 0.001, 0.0001, 8]
bash_func(run_settings, paras)

run_settings = [10, 3000, 300, 0.001, 0.0001, 0.996]
#BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY
paras = [10000, 16, 0.95, 0.001, 0.0001, 8]
bash_func(run_settings, paras)


