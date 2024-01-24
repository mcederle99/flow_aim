import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, ego_state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.gru = nn.GRUCell(ego_state_dim, 256)
        self.fc1 = nn.Linear(256 + ego_state_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.pi = nn.Linear(256, action_dim)

        self.max_action = max_action
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(self.device)
        
    def forward_train(self, state):
        """
        state is a list containing the elements in the batch: each element is in turn a list,
        corresponding to the state dimension in that timestep (15 * num_veh)
        """
        rnn_out = [] # this list will contain all the batch elements after the recursive layer (B, 270)
        for i in range(len(state)): # for every element in the batch
            current_state = torch.tensor(state[i], device=self.device, dtype=torch.float32) # take the corresponding state
            num_vehs = current_state.shape[0] // 15 # compute the number of vehicles in that instant
            current_state = current_state.view(num_vehs, 15) # reshape (Num_Veh, 15)
            hx = torch.zeros(256, device=self.device, dtype=torch.float32) # initialize
            for j in range(num_vehs): # for every vehicle present in that instant
                hx = self.gru(current_state[j], hx) # iterate in the LSTM cell
            hx = torch.cat((current_state[0], hx)) # concatenate the ego_state with the final LSTM output
            rnn_out.append(hx) # compose the final list
        x = torch.stack(rnn_out, dim=0) # transform it into a tensor
        
        # MLP part
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        #out = self.max_action * torch.sigmoid(self.pi(x))
        out = torch.tanh(self.pi(x))
        
        return out

    def forward(self, state):
        """
        state is a (V, F*V) tensor
        """
        num_vehs = state.shape[0] // 15 # compute the number of vehicles in that instant
        if num_vehs == 0:
            return torch.tensor([], device=self.device)
        
        rnn_out = []
        #for i in range(state.shape[0]): # for every vehicle
        current_state = state.view(num_vehs, 15) # reshape (Num_Veh, 15)
        hx = torch.zeros(256, device=self.device, dtype=torch.float32) # initialize
        for j in range(num_vehs): # for every vehicle present in that instant
            hx = self.gru(current_state[j], hx) # iterate in the LSTM cell
        hx = torch.cat((current_state[0], hx)) # concatenate the ego_state with the final LSTM output
        rnn_out.append(hx) # compose the final list
        x = torch.stack(rnn_out, dim=0) # transform it into a tensor

        # MLP part
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        #out = self.max_action * torch.sigmoid(self.pi(x))
        out = torch.tanh(self.pi(x))
        
        return out
    
class Critic(nn.Module):
    def __init__(self, ego_state_dim, action_dim):
        super(Critic, self).__init__()

        self.gru_1 = nn.GRUCell(ego_state_dim, 256)
        self.fc1_1 = nn.Linear(256 + ego_state_dim + action_dim, 1024)
        self.fc2_1 = nn.Linear(1024, 1024)
        self.fc3_1 = nn.Linear(1024, 512)
        self.fc4_1 = nn.Linear(512, 256)
        self.q_1 = nn.Linear(256, 1)

        self.gru_2 = nn.GRUCell(ego_state_dim, 256)
        self.fc1_2 = nn.Linear(256 + ego_state_dim + action_dim, 1024)
        self.fc2_2 = nn.Linear(1024, 1024)
        self.fc3_2 = nn.Linear(1024, 512)
        self.fc4_2 = nn.Linear(512, 256)
        self.q_2 = nn.Linear(256, 1)

        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(self.device)
        
    def forward(self, state, action):
        """
        - state is a list containing the elements in the batch: each element is in turn a list,
        corresponding to the state dimension in that timestep (15 * num_veh)
        - action is a tensor of dimension (B) containing the actions of all the vehicles in the batch
        """
        rnn_out_1 = [] # this list will contain all the batch elements after the recursive layer (B, 271)
        rnn_out_2 = [] # this list will contain all the batch elements after the recursive layer (B, 271)
        
        for i in range(len(state)): # for every element in the batch
            current_state = torch.tensor(state[i], device=self.device, dtype=torch.float32) # take the corresponding state
            num_vehs = current_state.shape[0] // 15 # compute the number of vehicles in that instant
            current_state = current_state.view(num_vehs, 15) # reshape (Num_Veh, 15)

            hx_1 = torch.zeros(256, device=self.device, dtype=torch.float32) # initialize
            hx_2 = torch.zeros(256, device=self.device, dtype=torch.float32) # initialize

            for j in range(num_vehs): # for every vehicle present in that instant

                hx_1 = self.gru_1(current_state[j], hx_1) # iterate in the LSTM cell
                hx_2 = self.gru_2(current_state[j], hx_2) # iterate in the LSTM cell
            
            aug_state = torch.cat((current_state[0], action[i]))
            
            hx_1 = torch.cat((aug_state, hx_1)) # concatenate the ego_state with the final LSTM output
            hx_2 = torch.cat((aug_state, hx_2)) # concatenate the ego_state with the final LSTM output
            
            rnn_out_1.append(hx_1) # compose the final list
            rnn_out_2.append(hx_2) # compose the final list

        x_1 = torch.stack(rnn_out_1, dim=0) # transform it into a tensor
        x_2 = torch.stack(rnn_out_2, dim=0) # transform it into a tensor
        
        # MLP part 1
        x_1 = F.relu(self.fc1_1(x_1))
        x_1 = F.relu(self.fc2_1(x_1))
        x_1 = F.relu(self.fc3_1(x_1))
        x_1 = F.relu(self.fc4_1(x_1))

        # MLP part 2
        x_2 = F.relu(self.fc1_2(x_2))
        x_2 = F.relu(self.fc2_2(x_2))
        x_2 = F.relu(self.fc3_2(x_2))
        x_2 = F.relu(self.fc4_2(x_2))
        
        out_1 = self.q_1(x_1)
        out_2 = self.q_2(x_2)
        
        return out_1, out_2
    
    def Q1(self, state, action):
        
        rnn_out_1 = [] # this list will contain all the batch elements after the recursive layer (B, 271)
        
        for i in range(len(state)): # for every element in the batch
            current_state = torch.tensor(state[i], device=self.device, dtype=torch.float32) # take the corresponding state
            num_vehs = current_state.shape[0] // 15 # compute the number of vehicles in that instant
            current_state = current_state.view(num_vehs, 15) # reshape (Num_Veh, 15)

            hx_1 = torch.zeros(256, device=self.device, dtype=torch.float32) # initialize

            for j in range(num_vehs): # for every vehicle present in that instant

                hx_1 = self.gru_1(current_state[j], hx_1) # iterate in the LSTM cell
            
            aug_state = torch.cat((current_state[0], action[i]))
            
            hx_1 = torch.cat((aug_state, hx_1)) # concatenate the ego_state with the final LSTM output
            
            rnn_out_1.append(hx_1) # compose the final list

        x_1 = torch.stack(rnn_out_1, dim=0) # transform it into a tensor
        
        # MLP part
        x_1 = F.relu(self.fc1_1(x_1))
        x_1 = F.relu(self.fc2_1(x_1))
        x_1 = F.relu(self.fc3_1(x_1))
        x_1 = F.relu(self.fc4_1(x_1))
        
        out_1 = self.q_1(x_1)
        
        return out_1
