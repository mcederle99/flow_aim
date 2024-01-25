import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, ego_state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.lstm1 = nn.LSTMCell(ego_state_dim, 256)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.lstm2 = nn.LSTMCell(256 + ego_state_dim, 256)
        self.pi = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(self.device)
        
    def forward(self, state):
        """
        state is a (V, F) tensor
        """
        batch_size = state.shape[0]
        num_vehs = state.shape[1] // 15
        if num_vehs == 0:
            return torch.tensor([], device=self.device)
        
        hx = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        for i in range(num_vehs): # for every vehicle in the scene
            current_state = state[:, 15*i:15*(i+1)]
            hx, cx = self.lstm1(current_state, (hx, cx)) # iterate in the LSTM cell
        # now hx is a 256-dim tensor
        
        # MLP part
        x = F.relu(self.fc1(hx))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        hx = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        actions = []
        for i in range(num_vehs):
            act = torch.cat((state[:, 15*i:15*(i+1)], x), dim=1)
            hx, cx = self.lstm2(act, (hx, cx))
            out = torch.tanh(self.pi(hx))
            actions.append(out)
        actions = torch.stack(actions, dim=1)
        
        if actions.squeeze().shape == torch.Size([]):
            return actions.squeeze().unsqueeze(dim=0)
        else:
            return actions.squeeze()
    
class Critic(nn.Module):
    def __init__(self, ego_state_dim, action_dim):
        super(Critic, self).__init__()

        self.lstm1_q1 = nn.LSTMCell(ego_state_dim + action_dim, 256)
        self.fc1_q1 = nn.Linear(256, 1024)
        self.fc2_q1 = nn.Linear(1024, 1024)
        self.fc3_q1 = nn.Linear(1024, 512)
        self.fc4_q1 = nn.Linear(512, 256)
        self.lstm2_q1 = nn.LSTMCell(256 + ego_state_dim + action_dim, 256)
        self.q_q1 = nn.Linear(256, 1)
        
        self.lstm1_q2 = nn.LSTMCell(ego_state_dim + action_dim, 256)
        self.fc1_q2 = nn.Linear(256, 1024)
        self.fc2_q2 = nn.Linear(1024, 1024)
        self.fc3_q2 = nn.Linear(1024, 512)
        self.fc4_q2 = nn.Linear(512, 256)
        self.lstm2_q2 = nn.LSTMCell(256 + ego_state_dim + action_dim, 256)
        self.q_q2 = nn.Linear(256, 1)


        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        self.to(self.device)
                
    def forward(self, state, action):
        """
        state is a (B, V*F) tensor
        action is a (B, V) tensor
        """
        batch_size = state.shape[0]
        num_vehs = state.shape[1] // 15
        if num_vehs == 0:
            return torch.tensor([], device=self.device)
        
        hx_q1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx_q1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        for i in range(num_vehs): # for every vehicle in the scene
            current_state = state[:, 15*i:15*(i+1)]
            hx_q1, cx_q1 = self.lstm1_q1(torch.cat((current_state, action[:, i].unsqueeze(dim=1)), dim=1), (hx_q1, cx_q1)) # iterate in the LSTM cell
        # now hx is a 256-dim tensor
        
        # MLP part
        x_q1 = F.relu(self.fc1_q1(hx_q1))
        x_q1 = F.relu(self.fc2_q1(x_q1))
        x_q1 = F.relu(self.fc3_q1(x_q1))
        x_q1 = F.relu(self.fc4_q1(x_q1))
        
        hx_q1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx_q1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        q_values_q1 = []
        for i in range(num_vehs):
            act_q1 = torch.cat((state[:, 15*i:15*(i+1)], action[:, i].unsqueeze(dim=1), x_q1), dim=1)
            hx_q1, cx_q1 = self.lstm2_q1(act_q1, (hx_q1, cx_q1))
            out_q1 = self.q_q1(hx_q1)
            q_values_q1.append(out_q1)
        q_values_q1 = torch.stack(q_values_q1, dim=1)

        #############################################################################################
        
        hx_q2 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx_q2 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        for i in range(num_vehs): # for every vehicle in the scene
            current_state = state[:, 15*i:15*(i+1)]
            hx_q2, cx_q2 = self.lstm1_q2(torch.cat((current_state, action[:, i].unsqueeze(dim=1)), dim=1), (hx_q2, cx_q2)) # iterate in the LSTM cell
        # now hx is a 256-dim tensor
        
        # MLP part
        x_q2 = F.relu(self.fc1_q2(hx_q2))
        x_q2 = F.relu(self.fc2_q2(x_q2))
        x_q2 = F.relu(self.fc3_q2(x_q2))
        x_q2 = F.relu(self.fc4_q2(x_q2))
        
        hx_q2 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx_q2 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        q_values_q2 = []
        for i in range(num_vehs):
            act_q2 = torch.cat((state[:, 15*i:15*(i+1)], action[:, i].unsqueeze(dim=1), x_q2), dim=1)
            hx_q2, cx_q2 = self.lstm2_q2(act_q2, (hx_q2, cx_q2))
            out_q2 = self.q_q2(hx_q2)
            q_values_q2.append(out_q2)
        q_values_q2 = torch.stack(q_values_q2, dim=1)

        
        return q_values_q1.squeeze(), q_values_q2.squeeze()
    
    def Q1(self, state, action):
        """
        state is a (B, V*F) tensor
        action is a (B, V) tensor
        """
        batch_size = state.shape[0]
        num_vehs = state.shape[1] // 15
        if num_vehs == 0:
            return torch.tensor([], device=self.device)
        
        hx_q1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx_q1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        for i in range(num_vehs): # for every vehicle in the scene
            current_state = state[:, 15*i:15*(i+1)]
            hx_q1, cx_q1 = self.lstm1_q1(torch.cat((current_state, action[:, i].unsqueeze(dim=1)), dim=1), (hx_q1, cx_q1)) # iterate in the LSTM cell
        # now hx is a 256-dim tensor
        
        # MLP part
        x_q1 = F.relu(self.fc1_q1(hx_q1))
        x_q1 = F.relu(self.fc2_q1(x_q1))
        x_q1 = F.relu(self.fc3_q1(x_q1))
        x_q1 = F.relu(self.fc4_q1(x_q1))
        
        hx_q1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx_q1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        q_values_q1 = []
        for i in range(num_vehs):
            act_q1 = torch.cat((state[:, 15*i:15*(i+1)], action[:, i].unsqueeze(dim=1), x_q1), dim=1)
            hx_q1, cx_q1 = self.lstm2_q1(act_q1, (hx_q1, cx_q1))
            out_q1 = self.q_q1(hx_q1)
            q_values_q1.append(out_q1)
        q_values_q1 = torch.stack(q_values_q1, dim=1)
        
        return q_values_q1.squeeze()
