from collections import deque
import torch
import numpy as np
import matplotlib.pyplot as plt
import util

import env as enviroment
#import agent
import baseline_agent
import time

#agent = agent.Fully_Agent(state_size=4, action_size=4, dinotype_num=8, seed=0)
#agent = agent.Q_Agent_Neighbour(state_size=6, action_size=4, dinotype_num=8, seed=0)
#agent = agent.Q_Agent_Vector(state_size=15, action_size=4, dinotype_num=8, seed=0)
#agent = agent.Q_Agent_Vector(state_size=15, action_size=4, dinotype_num=8, seed=0)
#agent = agent.Q_Agent_single_prob(state_size=5, action_size=4, dinotype_num=8, seed=0)

def dqn(n_runs, n_episodes, max_t=300, eps_start=1, eps_end = 1e-4, eps_decay=0.996):
    steps = np.zeros(n_episodes)
    acc_rewards = np.zeros(n_episodes)
    # scores = []

    # map_vec = env.init_map_vec()
    # probMap = np.full((8, 8), 0)
    # for num in map_vec:
    #     loc = util.num_to_loc(num,8)
    #     probMap[loc[0]][loc[1]] = 1

    # print(agent.probMap)

    for i_run in range(n_runs):
        # train
        print("run: ",i_run)
        # reset for each run
        agent = baseline_agent.Fully_Agent_Neighbour(state_size=6, action_size=4,
                                                     dinotype_num=8, seed=0)
        env = enviroment.Environment(size=8)
        map_vec = env.init_map_vec()
        eps = eps_start

        # provide the learned map
        for i_episode in range(n_episodes):
            if i_episode%10 == 0:
                print("episode: ", i_episode)
            state = env.reset()
            score = 0
            #agent.probMap = probMap
            #agent.visitMap = np.full((8, 8), 0)
            for t in range(max_t):
                success = False
                action = agent.choose_act(state, eps, t)
                next_state, reward, done = env.step(action)
                # clip_reward = np.clip(reward, -1, 1)
                clip_reward = np.sign(reward)*np.log(1+np.absolute(reward))
                agent.step(state, action, clip_reward, next_state, done, False, True) # not update the map
                state = next_state
                eps = max(eps * eps_decay, eps_end)
                score += reward
                if done:
                    # env.render()
                    print("step: ",t," ep_score: ",score)
                    steps[i_episode] = steps[i_episode] + t
                    acc_rewards[i_episode] += score
                    success = True
                    break

            if not success:
                steps[i_episode] = steps[i_episode] + max_t
                acc_rewards[i_episode] += score

        del agent, env, map_vec

    return acc_rewards, steps # , agent.probMap

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
    n_runs = run_settings['n_runs']
    n_episodes = run_settings['n_episodes']

    baseline_agent.para_print()
    baseline_agent.para_setting(paras)
    baseline_agent.para_print()

    scores, steps = dqn(run_settings['n_runs'],run_settings['n_episodes'],
                                 run_settings['max_local_t'],run_settings['eps_start'],
                                 run_settings['eps_end'],run_settings['eps_decay'])

    steps = steps / n_runs
    scores = scores / n_runs
    print("average steps", sum(steps) / len(steps))
    print("average reward", sum(scores) / len(scores))

    # print("probMap",agent.probMap)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    avg_steps = util.running_mean(steps, 200)
    plt.plot(np.arange(len(steps) - 200 + 1), avg_steps)
    plt.ylabel('Steps')
    plt.xlabel('# of Episodes')
    # plt.show()
    path = str(run_settings)+str(paras)+"steps"
    plt.savefig(path+'.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    avg_scores = util.running_mean(scores, 200)
    plt.plot(np.arange(len(scores) - 200 + 1), avg_scores)
    plt.ylabel('Reward')
    plt.xlabel('# of Episodes')
    # plt.show()
    path = str(run_settings)+str(paras)+"reward"
    plt.savefig(path+'.png')

# n_runs, n_episodes, max_t, eps_start, eps_end, eps_decay
run_settings = {'n_runs':1, 'n_episodes':5000, 'max_local_t': 100,
                'eps_start': 1, 'eps_end': 0.1, 'eps_decay': 0.9999}
#BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY
paras = {'buffer_size': 10000, 'batch_size': 16, 'gamma': 0.99,
         'tau': 0.001, 'lr': 0.0001, 'update_targetnet': 4}
bash_func(run_settings, paras)

# run_settings = [10, 3000, 300, 0.001, 0.0001, 0.996]
# #BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY
# paras = [10000, 16, 0.95, 0.001, 0.0001, 8]
# bash_func(run_settings, paras)
