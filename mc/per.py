import random
import numpy as np

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
            'obs': [],
            'action': np.zeros((capacity, 1)),
            'reward': np.zeros((capacity, 1)),
            'next_obs': [],
            'not_done': np.zeros((capacity, 1)),
        }
        
        self.next_idx = 0
        self.size = 0
        
    def add(self, obs, action, reward, next_obs, done):
        
        idx = self.next_idx
        
        self.data['obs'].append(obs.detach().cpu().tolist())
        self.data['action'][idx] = action.numpy()
        self.data['reward'][idx] = reward.numpy()
        self.data['next_obs'].append(next_obs.detach().cpu().tolist())
        self.data['not_done'][idx] = 1. - done.numpy()
        
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
            
        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-self.beta)
        
        for i in range(batch_size):
            idx = samples['indexes'][i]
            
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-self.beta)
            
            samples['weights'][i] = weight / max_weight
            
        for k, v in self.data.items():
            if k == 'obs':
                samples['obs'] = []
                for i in samples['indexes']:
                    samples['obs'].append(self.data['obs'][i])
            elif k == 'next_obs':
                samples['next_obs'] = []
                for i in samples['indexes']:
                    samples['next_obs'].append(self.data['next_obs'][i])
            else:
                samples[k] = v[samples['indexes']]
            
        return samples
    
    def update_priorities(self, indexes, priorities):
        
        for idx, priority in zip(indexes, priorities):
            
            self.max_priority = max(self.max_priority, priority)
            
            priority_alpha = priority ** self.alpha
            
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)
            
    def is_full(self):
        return self.capacity == self.size
