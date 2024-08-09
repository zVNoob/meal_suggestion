import random
import torch

#Source: https://github.com/tsmatz/reinforcement-learning-tutorials/blob/master/01-dqn.ipynb


class replayMemory:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "xpu" if torch.xpu.is_available() else 
                                   "cpu")

    def add(self, item):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(item)

    def sample(self):
        # sampling
        items = random.sample(self.buffer, len(self.buffer)//3)
        # divide each columns
        states   = [i[0] for i in items]
        actions  = [i[1] for i in items]
        rewards  = [i[2] for i in items]
        n_states = [i[3] for i in items]
        dones    = [i[4] for i in items]
        # convert to tensor
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        n_states = torch.tensor(n_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        #states = torch.reshape(states, (-1, batch_size, 4))
        # actions = torch.reshape(actions, (-1, batch_size))
        # rewards = torch.reshape(rewards, (-1, batch_size))
        # n_states = torch.reshape(n_states, (-1, batch_size, 4))
        # dones = torch.reshape(dones, (-1, batch_size))
        # return result
        return states, actions, rewards, n_states, dones

    def length(self):
        return len(self.buffer)
