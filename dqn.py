from torch import nn
from torch import optim
import torch
import torch.nn.functional as F
import numpy as np

#Source: https://github.com/tsmatz/reinforcement-learning-tutorials/blob/master/01-dqn.ipynb

class DQN(nn.Module):
    def __init__(self, input_size, output_size, source = None):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "xpu" if torch.xpu.is_available() else 
                                   "cpu")
        self.to(self.device)
        if source:
            self.load_state_dict(source.state_dict())
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x
    def optimize(self, target, states, actions, rewards, next_states, dones):
        #
        # Compute target
        #

        with torch.no_grad():
            # compute Q(s_{t+1})                               : size=[batch_size, 2]
            target_vals_for_all_actions = target(next_states.unsqueeze(dim=0))  
            # compute argmax_a Q(s_{t+1})                      : size=[batch_size]
            target_actions = torch.argmax(target_vals_for_all_actions, 1)
            # compute max Q(s_{t+1})                           : size=[batch_size]
            target_actions_one_hot = F.one_hot(target_actions, self.output.out_features).float()
            target_vals = torch.sum(target_vals_for_all_actions * target_actions_one_hot, 1)
            # compute r_t + gamma * (1 - d_t) * max Q(s_{t+1}) : size=[batch_size]
            target_vals_masked = (1.0 - dones) * target_vals
            q_vals1 = rewards + 0.99 * target_vals_masked

        self.optimizer.zero_grad()

        #
        # Compute q-value
        #
        actions_one_hot = F.one_hot(actions, self.output.out_features).float()
        q_vals2 = torch.sum(self(states).unsqueeze(dim=0) * actions_one_hot, 1)

        #
        # Get MSE loss and optimize
        #
        loss = F.mse_loss(
            q_vals1.detach(),
            q_vals2,
            reduction="mean")
        loss.backward()
        self.optimizer.step()

    def pick_sample(self,state, epsilon):
        with torch.no_grad():
        # get optimal action,
        # but with greedy exploration (to prevent picking up same values in the first stage)
            if np.random.random() > epsilon:
                s_batch = state.clone().detach()
                s_batch = s_batch.unsqueeze(dim=0)  # to make batch with size=1
                q_vals_for_all_actions = self(s_batch)
                a = torch.argmax(q_vals_for_all_actions, 1)
                a = a.squeeze(dim=0)
                a = a.item()
            else:
                a = np.random.randint(0, self.output.out_features)
        return a

    def evaluate(self,env,epsilon = 0.0,memory = None):
        with torch.no_grad():
            state = env.reset()
            done = False
            total = 0
            while not done:
                a = self.pick_sample(state, epsilon)
                s_next, reward, done = env.step(a)
                if memory:
                    memory.add([state.tolist(), a, reward, s_next.tolist(), float(done)])
                total += reward
                state = s_next
        return total

