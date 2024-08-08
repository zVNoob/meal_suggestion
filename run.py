import os
os.environ['PATH'] = os.getcwd() + os.pathsep + os.environ['PATH']

import dqn
import env
import torch

money = input("Enter the money you have: ")


Environment = env.Env(100000, 3, 4)
Network = dqn.DQN_Navie(Environment.size()[0], Environment.size()[1])

Network.load_state_dict(torch.load('model.pth'))

done = False
input = Environment.reset()
while not done:
    output = Network(input)
    input, reward, done = Environment.step(output)
