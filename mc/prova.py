import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, ego_state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.lstm_1 = nn.LSTMCell(ego_state_dim, 256)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.lstm_2 = nn.LSTMCell(256, 256)
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
            cx = torch.zeros(256, device=self.device, dtype=torch.float32) # initialize
            for j in range(num_vehs): # for every vehicle present in that instant
                hx, cx = self.lstm_1(current_state[j], (hx, cx)) # iterate in the LSTM cell
            rnn_out.append(hx) # compose the final list
        x = torch.stack(rnn_out, dim=0) # transform it into a tensor

        # MLP part
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # x is now (B, 256)
        actions_batch = []
        for i in range(x.shape[0]):
            actions = []
            hx = torch.zeros(256, device=self.device, dtype=torch.float32)
            cx = torch.zeros(256, device=self.device, dtype=torch.float32)
            for j in range(len(state[i])//15):
                #print(len(state[i]))
                hx, cx = self.lstm_2(x[i,:], (hx, cx))
                out = torch.tanh(self.pi(cx))
                actions.append(out)
            actions = torch.stack(actions, dim=0)
            actions_batch.append(actions)

        #actions_batch = torch.stack(actions_batch, dim=0)
        return actions_batch

def generate_state(nv):
    statef = torch.tensor([])
    for i in range(nv):
        state1 = torch.randn((5,), dtype=torch.float32)
        state2 = F.one_hot(torch.arange(3), num_classes=3).float()
        random2 = np.random.choice(3)
        state2 = state2[random2]
        state3 = F.one_hot(torch.arange(3), num_classes=3).float()
        random3 = np.random.choice(3)
        state3 = state3[random3]
        state4 = F.one_hot(torch.arange(4), num_classes=4).float()
        random4 = np.random.choice(4)
        state4 = state4[random4]
        state = torch.cat((state1, state2, state3, state4))
        statef = torch.cat((statef, state))
    return statef
batch = []
for i in range(1, 5):
    s = generate_state(i)
    batch.append(s)

model = Actor(15, 1, 1)
act = model.forward_train(batch)
print(act)
