import copy
import torch
import torch.nn.functional as f
from torch_geometric.loader import DataLoader
import torch_geometric.nn as gnn
import numpy as np
from nnetworks import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent(object):
    def __init__(
            self,
            state_dim,
            edge_dim,
            action_dim,
            discount=0.99,
            tau=0.005,
            epsilon=1.0,
            eps_min=0.01,
            eps_dec=5e-7,
    ):

        self.q_eval = DQN(state_dim, edge_dim, action_dim).to(device)
        self.q_next = copy.deepcopy(self.q_eval)
        self.q_optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(5)]

        self.total_it = 0

    def select_action(self, state, edge_index, edge_attr, edge_type, evaluate=False):
        if (np.random.random() > self.epsilon) or evaluate:
            _, advantage = self.q_eval(state, edge_index, edge_attr, edge_type)
            action = torch.argmax(advantage.cpu(), dim=1)
            return action.numpy()
        else:
            action = np.random.choice(self.action_space, state.shape[0], p=[0.1, 0.1, 0.1, 0.35, 0.35])
            return action

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        b = replay_buffer.sample(batch_size)
        loader = DataLoader(b, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
        batch = next(iter(loader))
        indices = np.arange(batch_size)

        with torch.no_grad():
            v_s_eval, a_s_eval = self.q_eval(batch.x_t, batch.edge_index_t, batch.edge_attr_t, batch.edge_type_t)
            q_eval = torch.add(v_s_eval, (a_s_eval - a_s_eval.mean(dim=1, keepdim=True)))
            max_actions = torch.argmax(q_eval, dim=1)

            v_s_, a_s_ = self.q_next(batch.x_t, batch.edge_index_t, batch.edge_attr_t, batch.edge_type_t)
            q_next = torch.add(v_s_, (a_s_ - a_s_.mean(dim=1, keepdim=True)))

            q_next = q_next[np.arange(q_next.shape[0]), max_actions]
            q_next = gnn.global_mean_pool(q_next, batch.x_t_batch)  # careful with this

            q_target = batch.reward.unsqueeze(dim=1) + batch.not_done.unsqueeze(dim=1) * self.discount * q_next

        v_s, a_s = self.q_eval(batch.x_s, batch.edge_index_s, batch.edge_attr_s, batch.edge_type_s)
        q_predict = torch.add(v_s, (a_s - a_s.mean(dim=1, keepdim=True)))

        q_predict = q_predict[indices, batch.actions]
        q_predict = gnn.global_mean_pool(q_predict, batch.x_s_batch)  # careful with this

        loss = f.mse_loss(q_target, q_predict)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        if self.total_it % 2 == 0:
            # Update the frozen target model
            for param, target_param in zip(self.q_eval.parameters(), self.q_next.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.q_eval.state_dict(), filename + "_q_eval")
        torch.save(self.q_optimizer.state_dict(), filename + "_q_optimizer")

    def load(self, filename):
        self.q_eval.load_state_dict(torch.load(filename + "_q_eval", map_location=torch.device(device)))
        self.q_optimizer.load_state_dict(torch.load(filename + "_q_optimizer"))
        self.q_next = copy.deepcopy(self.q_eval)
