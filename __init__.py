#!/usr/bin/python3
#i hate python for this
import os
os.environ['PATH'] = os.getcwd() + os.pathsep + os.environ['PATH']

import dqn
import env
import os
import sys

os.set_blocking(sys.stdin.fileno(), False)

Environment = env.Env(100000, 3, 4)
Network = dqn.DQN_Navie(Environment.size()[0], Environment.size()[1])

while 1:
    input = Environment.reset()
    done = False
    while not done:
        output = Network(input)
        input, reward, done = Environment.step(output)
        Network.optimize(Network.discounted_reward(reward))
    try:
        os.read(0, 1)
        break
    except:
        pass



