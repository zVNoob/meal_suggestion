#!/usr/bin/python3
#i hate python for this
import os
os.environ['PATH'] = os.getcwd() + os.pathsep + os.environ['PATH']

import dqn
import env
import os
import sys
import torch

clear = lambda: os.system('cls') if os.name == 'win32' else os.system('clear')

os.set_blocking(sys.stdin.fileno(), False)

Environment = env.Env(100000, 3, 4)
Network = dqn.DQN(Environment.size()[0], Environment.size()[1])

while 1:
    input = Environment.reset()
    done = False
    rewards = []
    log_probs = []
    while not done:
        probs = Network(input)
        temp = torch.distributions.Categorical(probs)
        output = temp.sample()
        input, reward, done = Environment.step(output.item())
        log_probs.append(temp.log_prob(output))
        rewards.append(reward)

    Network.optimize(log_probs,rewards)
    clear()
    print(sum(rewards))
    try:
        os.read(sys.stdin.fileno(), 1)
        break
    except:
        pass

torch.save(Network.state_dict(), 'model.pth')



