from torch import Tensor, nn
from torch import optim
import torch
import torch.nn.functional as F
import numpy as np

#Source: https://www.geeksforgeeks.org/reinforcement-learning-using-pytorch/

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=-1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    def _compute_discounted_rewards(self,rewards:list, gamma=0.99):
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        return discounted_rewards

    def optimize(self,log_probs:list,rewards:list):
        discounted_rewards = self._compute_discounted_rewards(rewards)
        policy_loss = list[Tensor]()
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * Gt)
        self.optimizer.zero_grad()
        policy_loss = torch.cat([i.reshape(1) for i in policy_loss]).sum()
        policy_loss.backward()
        self.optimizer.step()
