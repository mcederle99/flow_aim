import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rnn = nn.LSTMCell(10, 20)  # (input_size, hidden_size)
inpu = torch.randn(2, 3, 10)  # (time_steps, batch, input_size)
hx = torch.randn(3, 20)  # (batch, hidden_size)
cx = torch.randn(3, 20)
output = []
for i in range(inpu.size()[0]):
    hx, cx = rnn(inpu[i], (hx, cx))
    output.append(hx)
output = torch.stack(output, dim=0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.lstm = nn.LSTMCell(state_dim, 256)
        self.fc1 = nn.Linear(256+state_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.pi = nn.Linear(256, action_dim)

        self.max_action = max_action


