#!/usr/bin/python3
#i hate python for this
import os
os.environ['PATH'] = os.getcwd() + os.pathsep + os.environ['PATH']

import dqn
import env
import replay
import os
import sys
import torch
import numpy

clear = lambda: os.system('cls') if os.name == 'win32' else os.system('clear')

os.set_blocking(sys.stdin.fileno(), False)

Environment = env.Env(100000, 3, 15)
Network = dqn.DQN(Environment.size()[0], Environment.size()[1])
Target = dqn.DQN(Environment.size()[0], Environment.size()[1],Network)
Memory = replay.replayMemory(10000)

epsilon = 1.0
epsilon_decay = epsilon / 10000
epsilon_final = 0.1


count = 0

reward_record = []

while 1:
    # show current output
    clear()
    total_reward = Network.evaluate(Environment)
    if len(reward_record) > 200:
        reward_record.pop(0)
    reward_record.append(total_reward)
    print(total_reward)
    # training
    while Memory.length() < 2000:
        Network.evaluate(Environment, epsilon, Memory)
    for _ in range(500):
        Network.evaluate(Environment, epsilon, Memory)
    states, actions, rewards, n_states, dones = Memory.sample()
    for j in range(len(actions)):
        Network.optimize(Target,states[j], actions[j], rewards[j], n_states[j], dones[j])
    if epsilon - epsilon_decay >= epsilon_final:
        epsilon -= epsilon_decay
    count += 1
    if count % 100 == 0:
        Target.load_state_dict(Network.state_dict())
        count = 0
    if numpy.average(reward_record) > 1000:
        reward_record = []
        Environment.initial_money += 100000
        if Environment.initial_money >= 5100000:
            break

    try:
        os.read(sys.stdin.fileno(), 1)
        break
    except:
        pass

torch.save(Network.state_dict(), 'model.pth')



