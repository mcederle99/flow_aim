import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(2**20)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = {}
        self.action = {}
        self.next_state = {}
        self.reward = {}
        self.not_done = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def add(self, state, action, next_state, reward, not_done):
        self.state[self.ptr] = state.detach().cpu()
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state.detach().cpu()
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = not_done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        max_num_vehs = 0
        for i in range(len(ind)):
            num_vehs = self.state[ind[i]].shape[0]
            if num_vehs > max_num_vehs:
                max_num_vehs = num_vehs
        max_num_vehs = max_num_vehs // 12
        
        batch_state = torch.zeros((batch_size, 12*max_num_vehs), dtype=torch.float32)
        batch_action = torch.zeros((batch_size, max_num_vehs), dtype=torch.float32)
        batch_reward = torch.zeros(batch_size, dtype=torch.float32)
        batch_next_state = torch.zeros((batch_size, 12*max_num_vehs), dtype=torch.float32)
        batch_not_done = torch.zeros(batch_size)
        
        mask = torch.ones((batch_size, max_num_vehs), dtype=torch.float32)
        
        for i in range(len(ind)):
            state = self.state[ind[i]]
            action = self.action[ind[i]]
            next_state = self.next_state[ind[i]]
            if next_state.shape[0] > state.shape[0]:
                diff = next_state.shape[0] - state.shape[0]
                next_state = next_state[:-diff]

            num_padding = max_num_vehs - (state.shape[0] // 12)
            for j in range(num_padding):
                state = torch.cat((state, torch.zeros(12, dtype=torch.float32)))
                action = torch.cat((action, torch.tensor([0.0])))
                next_state = torch.cat((next_state, torch.zeros(12, dtype=torch.float32)))
                
                mask[i, -(j+1)] = 0
        
            batch_state[i] = state
            batch_action[i] = action
            batch_reward[i] = self.reward[ind[i]]
            batch_next_state[i] = next_state
            batch_not_done[i] = self.not_done[ind[i]]
        
        batch_state = batch_state.view(batch_size, max_num_vehs, 12)
        batch_next_state = batch_next_state.view(batch_size, max_num_vehs, 12)
        
        return (
            batch_state.to(self.device),
            batch_action.to(self.device),
            batch_next_state.to(self.device),
            batch_reward.to(self.device),
            batch_not_done.to(self.device),
            mask.to(self.device)
        )
