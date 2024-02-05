import numpy as np
import torch
from network import Qfunc
from utils import map_actions

state = torch.ones(15)
state = state.view(1,-1,15)

q = Qfunc()
actions = q(state)
action = torch.argmax(actions, -1).squeeze()

print(action.unsqueeze(dim=0))
