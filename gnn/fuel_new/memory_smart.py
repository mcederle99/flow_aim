import numpy as np
import torch
from torch_geometric.data import Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6)):
        self.buffer = []
        self.buffer_size = max_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done, omega):
        del state.dist

        transition = PairData(x_s=state.x, edge_index_s=state.edge_index,
                              edge_attr_s=state.edge_attr, edge_type_s=state.edge_type,
                              x_t=next_state.x, edge_index_t=next_state.edge_index,
                              edge_attr_t=next_state.edge_attr, edge_type_t=next_state.edge_type)

        transition.actions = torch.tensor(action, device=device).unsqueeze(dim=1)
        transition.reward = torch.tensor([reward], dtype=torch.float32, device=device)
        transition.not_done = 1. - torch.tensor([done], device=device)
        transition.omega_s = torch.tensor([[omega, 1 - omega]], dtype=torch.float, device=device).repeat(state.x.shape[0], 1)
        transition.omega_t = torch.tensor([[omega, 1 - omega]], dtype=torch.float, device=device).repeat(
            next_state.x.shape[0], 1)

        if len(list(self.buffer)) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = []
        for i in ind:
            batch.append(self.buffer[i].to(self.device))

        return batch
