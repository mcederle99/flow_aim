import random
import numpy as np
import torch

class PrioritizedReplayBuffer(object):
    
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity # we use a power of 2 for capacity because it simplifies the code
        self.alpha = alpha
        self.beta = beta  # importance-sampling, from initial value increasing to 1, often 0.4
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.beta_increment_per_sampling = 1e-4  # annealing the bias, often 1e-3
        
        # maintain segment binary trees to take sum and find minimum over a range
        self.priority_sum = [0 for _ in range(2*self.capacity)]
        self.priority_min = [float('inf') for _ in range(2*self.capacity)]
        
        self.max_priority = 1. # current max priority to be assigned to new transitions
        
        self.data = {
            'obs': {},
            'action': {},
            'reward': {},
            'next_obs': {},
            'not_done': {},
        }
        
        self.next_idx = 0
        self.size = 0
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add(self, obs, action, reward, next_obs, done):
        
        idx = self.next_idx
        
        self.data['obs'][idx] = obs.detach().cpu()
        self.data['action'][idx] = action.detach().cpu()
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs.detach().cpu()
        self.data['not_done'][idx] = 1. - done
        
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        
        priority_alpha = self.max_priority ** self.alpha # new samples get max_priority
        
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)
        
    # set priority in binary segment tree for minimum
    def _set_priority_min(self, idx, priority_alpha):
        
        idx += self.capacity # leaf of the binary tree
        self.priority_min[idx] = priority_alpha
        
        while idx >= 2: # update tree by traversing along ancestors, continue until the root of the tree
            idx //= 2 # get the index of the parent node
            self.priority_min[idx] = min(self.priority_min[2*idx], self.priority_min[2*idx + 1]) # value of the
            # parent node is the minimum of its two children
            
    # set priority in binary segment tree for sum
    def _set_priority_sum(self, idx, priority):
        
        idx += self.capacity # leaf of the binary tree
        self.priority_sum[idx] = priority
        
        while idx >= 2: # update tree by traversing along ancestors, continue until the root of the tree
            idx //= 2 # get the index of the parent node
            self.priority_sum[idx] = self.priority_sum[2*idx] + self.priority_sum[2*idx + 1] # value of the
            # parent node is the sum of its two children
            
    def _sum(self):
        return self.priority_sum[1] # the root node keeps the sum of all values
    
    def _min(self):
        return self.priority_min[1] # the root node keeps the min of all values
    
    def find_prefix_sum_idx(self, prefix_sum):
        
        # start from the root
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx*2] >= prefix_sum: # if the sum of the left branch is higher than the required sum
                idx = 2*idx # go to the left branch of the tree
            else: # otherwise go to the right branch
                prefix_sum -= self.priority_sum[idx*2] # and reduce the sum of left branch from required sum
                idx = 2*idx + 1
                
        return idx - self.capacity # we are at the leaf node
    
    def sample(self, batch_size):
        
        # initialize samples
        samples = {
            'weights': np.zeros(shape=(batch_size), dtype=np.float32),
            'indexes': np.zeros(shape=(batch_size), dtype=np.int32)
        }
        
        self.beta = np.amin([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        
        # get sample indexes
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx
            
        max_num_vehs = 0
        for i in range(batch_size):
            idx = samples['indexes'][i]
            num_vehs = self.data['obs'][idx].shape[0]
            if num_vehs > max_num_vehs:
                max_num_vehs = num_vehs
        max_num_vehs = max_num_vehs // 15

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-self.beta)

        batch_state = torch.zeros((batch_size, 15*max_num_vehs), dtype=torch.float32)
        batch_action = torch.zeros((batch_size, 1), dtype=torch.float32)
        batch_reward = torch.zeros((batch_size, 1), dtype=torch.float32)
        batch_next_state = torch.zeros((batch_size, 15*max_num_vehs), dtype=torch.float32)
        batch_not_done = torch.zeros((batch_size, 1))

        mask = torch.ones((batch_size, max_num_vehs), dtype=torch.float32)

        for i in range(batch_size):
            idx = samples['indexes'][i]

            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-self.beta)
            samples['weights'][i] = weight / max_weight
            
            state = self.data['obs'][idx]
            action = self.data['action'][idx]
            reward = self.data['reward'][idx]
            next_state = self.data['next_obs'][idx]
            not_done = self.data['not_done'][idx]

            if next_state.shape[0] > state.shape[0]:
                diff = next_state.shape[0] - state.shape[0]
                next_state = next_state[:-diff]

            num_padding = max_num_vehs - (state.shape[0] // 15)
            for j in range(num_padding):
                state = torch.cat((state, torch.zeros(15, dtype=torch.float32)))
                #action = torch.cat((action, torch.tensor([0.0])))
                #reward = torch.cat((reward, torch.tensor([0.0])))
                next_state = torch.cat((next_state, torch.zeros(15, dtype=torch.float32)))
                #not_done = torch.cat((not_done, torch.tensor([0.0])))

                mask[i, -(j+1)] = 0

            batch_state[i] = state
            batch_action[i] = action
            batch_reward[i] = reward
            batch_next_state[i] = next_state
            batch_not_done[i] = not_done

        samples['obs'] = batch_state.to(self.device)
        samples['action'] = batch_action.to(self.device)
        samples['reward'] = batch_reward.to(self.device)
        samples['next_obs'] = batch_next_state.to(self.device)
        samples['not_done'] = batch_not_done.to(self.device)
        samples['mask'] = mask.to(self.device)

        return samples
    
    def update_priorities(self, indexes, priorities):
        
        for idx, priority in zip(indexes, priorities):
            
            self.max_priority = max(self.max_priority, priority)
            
            priority_alpha = priority ** self.alpha
            
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)
            
    def is_full(self):
        return self.capacity == self.size
