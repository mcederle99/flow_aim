import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, ego_state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.lstm1 = nn.LSTMCell(ego_state_dim, 256)
        self.lstm2 = nn.LSTMCell(ego_state_dim, 256)
        self.fc1 = nn.Linear(512 + ego_state_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.pi = nn.Linear(256, action_dim)

        self.max_action = max_action
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(self.device)
        
    def forward(self, state):
        """
        state is a (V, F*V) tensor
        """
        batch_size = state.shape[0]
        num_vehs = state.shape[1] // 15 # compute the number of vehicles in that instant
        if num_vehs == 0:
            return torch.tensor([], device=self.device)
        
        hx1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        hx2 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx2 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        for j in range(num_vehs): # for every vehicle present in that instant
            current_state = state[:, 15*j:15*(j+1)]
            hx1, cx1 = self.lstm1(current_state, (hx1, cx1)) # iterate in the LSTM cell
            hx2, cx2 = self.lstm2(current_state, (hx2, cx2)) # iterate in the LSTM cell
        hx = torch.cat((state[:, 0:15], hx1, hx2), dim=1) # concatenate the ego_state with the final LSTM output
        
        # MLP part
        x = F.relu(self.fc1(hx))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        actions = torch.tanh(self.pi(x))
        
        if actions.squeeze().shape == torch.Size([]):
            return actions.squeeze().unsqueeze(dim=0)
        else:
            return actions.squeeze()

class Critic(nn.Module):
    def __init__(self, ego_state_dim, action_dim):
        super(Critic, self).__init__()

        self.lstm1_1 = nn.LSTMCell(ego_state_dim, 256)
        self.lstm2_1 = nn.LSTMCell(ego_state_dim, 256)
        self.fc1_1 = nn.Linear(512 + ego_state_dim + action_dim, 1024)
        self.fc2_1 = nn.Linear(1024, 1024)
        self.fc3_1 = nn.Linear(1024, 512)
        self.fc4_1 = nn.Linear(512, 256)
        self.q_1 = nn.Linear(256, 1)

        self.lstm1_2 = nn.LSTMCell(ego_state_dim, 256)
        self.lstm2_2 = nn.LSTMCell(ego_state_dim, 256)
        self.fc1_2 = nn.Linear(512 + ego_state_dim + action_dim, 1024)
        self.fc2_2 = nn.Linear(1024, 1024)
        self.fc3_2 = nn.Linear(1024, 512)
        self.fc4_2 = nn.Linear(512, 256)
        self.q_2 = nn.Linear(256, 1)

        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(self.device)
        
    def forward(self, state, action):
        """
        - state is a (B, V*F) tensor 
        - action is a tensor of dimension (B) containing the actions of all the vehicles in the batch
        """
        batch_size = state.shape[0]
        num_vehs = state.shape[1] // 15

        hx1_1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx1_1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        hx1_2 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx1_2 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        hx2_1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx2_1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        hx2_2 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx2_2 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        for j in range(num_vehs): # for every vehicle present
            current_state = state[:, 15*j:15*(j+1)]
            hx1_1, cx1_1 = self.lstm1_1(current_state, (hx1_1, cx1_1)) # iterate in the LSTM cell
            hx1_2, cx1_2 = self.lstm1_2(current_state, (hx1_2, cx1_2)) # iterate in the LSTM cell
            hx2_1, cx2_1 = self.lstm2_1(current_state, (hx2_1, cx2_1)) # iterate in the LSTM cell
            hx2_2, cx2_2 = self.lstm2_2(current_state, (hx2_2, cx2_2)) # iterate in the LSTM cell
        
        hx_1 = torch.cat((state[:, 0:15], action.unsqueeze(dim=1), hx1_1, hx2_1), dim=1) # concatenate the ego_state with the final LSTM output
        hx_2 = torch.cat((state[:, 0:15], action.unsqueeze(dim=1), hx1_2, hx2_2), dim=1) # concatenate the ego_state with the final LSTM output
        
        # MLP part 1
        x_1 = F.relu(self.fc1_1(hx_1))
        x_1 = F.relu(self.fc2_1(x_1))
        x_1 = F.relu(self.fc3_1(x_1))
        x_1 = F.relu(self.fc4_1(x_1))

        # MLP part 2
        x_2 = F.relu(self.fc1_2(hx_2))
        x_2 = F.relu(self.fc2_2(x_2))
        x_2 = F.relu(self.fc3_2(x_2))
        x_2 = F.relu(self.fc4_2(x_2))
        
        out_1 = self.q_1(x_1)
        out_2 = self.q_2(x_2)
        
        return out_1.squeeze(), out_2.squeeze()
    
    def Q1(self, state, action):
        """
        state is a (B, V*F) tensor
        action is a tensor of dimension (B) containing the actions of all the vehicles in the batch
        """
        batch_size = state.shape[0]
        num_vehs = state.shape[1] // 15

        hx1_1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx1_1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)        
        hx2_1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)
        cx2_1 = torch.zeros((batch_size, 256), device=self.device, dtype=torch.float32)        
        for j in range(num_vehs): # for every vehicle present
            current_state = state[:, 15*j:15*(j+1)]
            hx1_1, cx1_1 = self.lstm1_1(current_state, (hx1_1, cx1_1)) # iterate in the LSTM cell
            hx2_1, cx2_1 = self.lstm2_1(current_state, (hx2_1, cx2_1)) # iterate in the LSTM cell
        
        hx_1 = torch.cat((state[:, 0:15], action.unsqueeze(dim=1), hx1_1, hx2_1), dim=1) # concatenate the ego_state with the final LSTM output
        
        # MLP part 1
        x_1 = F.relu(self.fc1_1(hx_1))
        x_1 = F.relu(self.fc2_1(x_1))
        x_1 = F.relu(self.fc3_1(x_1))
        x_1 = F.relu(self.fc4_1(x_1))

        out_1 = self.q_1(x_1)
        
        return out_1.squeeze()
