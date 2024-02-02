import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import Qfunc

# hyperparameters
batch_size = 128
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#------------

class DQN():
    def __init__(self, state_dim, action_dim, epsilon=1.0, discount=0.99, tau=0.005, eps_min=0.01, eps_dec=5e-7,
            replace=1000, filename=''):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.discount = discount
        self.tau = tau
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace = replace
        self.filename = filename

        self.q = Qfunc().to(device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)

        self.total_it = 0

    def select_action(self, state):
        num_vehs = state.shape[0] // 15 
        if np.random.random() > self.epsilon:
            state = state.view(1, -1, 15).to(device)
            actions = self.q(state)
            action = torch.argmax(actions, -1).squeeze()
        else:
            #num_vehs = state.shape[1] // 15
            action = torch.randint(0, 21, (num_vehs,))

        return action.detach().cpu()

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def train(self, replay_buffer, batch_size=batch_size):
        if replay_buffer.size < batch_size:
            return

        self.total_it += 1

        # sample replay buffer
        state, action, next_state, reward, not_done, mask = replay_buffer.sample(batch_size)

        state = state.view(batch_size, -1, 15)
        next_state = state.view(batch_size, -1, 15)
        action = F.one_hot(action.to(torch.long))
        
        q_pred = self.q(state)
        q_pred = (q_pred * action).sum(-1)
        q_pred = q_pred.masked_fill(mask == 0, 0.0)

        with torch.no_grad():
            q_next = self.q_target(next_state).max(dim=2)[0]
            q_next = reward + self.discount * q_next * not_done
            q_next = q_next.masked_fill(mask == 0, 0.0)

        loss = F.mse_loss(q_pred, q_next)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        self.decrement_epsilon()

        # Update the frozen target models
        if self.total_it % 2 == 0:
            for param, target_param in zip(self.q.parameters(), self.q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
            torch.save(self.q.state_dict(), filename + "_q")
            torch.save(self.q_optimizer.state_dict(), filename + "_q_optimizer")

    def load(self, filename):
            self.q.load_state_dict(torch.load(filename + "_q"))
            self.q_optimizer.load_state_dict(torch.load(filename + "_q_optimizer"))
            self.q_target = copy.deepcopy(self.q)
