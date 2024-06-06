import numpy as np

class ReplayBuffer:
    def __init__(self, size):
        
        self.size = size
        self.buffer = []
        self.index = 0
        self.length = 0
        
    def add(self, nodes, edges, edges_type, action, reward, nodes_, edges_, edges_type_, done):
        
        data = (nodes, edges, edges_type, action, reward, nodes_, edges_, edges_type_, done)
        
        if self.index >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data
            
        self.index = (self.index + 1) % self.size
        
        self.length = min(self.length + 1, self.size)
        
    def sample(self, batch_size, n_steps=1):
        
        samples = {'weights': np.ones(shape=batch_size, dtype=np.float32),
                   'indexes': np.random.choice(self.length - n_steps + 1, batch_size, replace=False)}
        
        sample_data = []
        if n_steps == 1:
            for i in samples['indexes']:
                data_i = self.buffer[i]
                sample_data.append(data_i)
        else:
            for i in samples['indexes']:
                data_i = self.buffer[i: i + n_steps]
                sample_data.append(data_i)
        return samples, sample_data
