import os
os.environ['PATH'] = os.getcwd() + os.pathsep + os.environ['PATH']

import dqn
import env
import torch

money = int(input("Enter the money you have: "))


Environment = env.Env(money, 3, 15)
Network = dqn.DQN(Environment.size()[0], Environment.size()[1])

Network.load_state_dict(torch.load('model.pth'))

Network.evaluate(Environment)
