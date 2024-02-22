import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks_discrete import Actor, Critic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.3,
        policy_freq=2,
        filename=''
    ):

        self.actor = Actor(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.filename = filename

        self.total_it = 0
#        self.mean_list = []
#        self.std_list = []


    def select_action(self, state):
        state = state.to(device)
        return self.actor(state).detach().cpu()
    
    def train(self, replay_buffer, replay_buffer_col, batch_size=256):
        
        self.total_it += 1
        
        batch_size_col = min(64, replay_buffer_col.size)
        batch_size -= batch_size_col

        # Sample replay buffer 
        state1, action1, next_state1, reward1, not_done1, mask1, max_num_vehs = replay_buffer.sample(batch_size)
        state2, action2, next_state2, reward2, not_done2, mask2, _ = replay_buffer_col.sample(batch_size_col, max_num_vehs)

        state = torch.cat((state1, state2), dim=0)
        action = torch.cat((action1, action2), dim=0)
        next_state = torch.cat((next_state1, next_state2), dim=0)
        reward = torch.cat((reward1, reward2), dim=0)
        not_done = torch.cat((not_done1, not_done2), dim=0)
        mask = torch.cat((mask1, mask2), dim=0)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
#            noise = (
#                torch.randn_like(action) * self.policy_noise
#            ).clamp(-self.noise_clip, self.noise_clip).squeeze()
#            next_action = (
#                self.actor_target(next_state) + noise
#            ).clamp(-1, 1)
#            next_action = next_action.masked_fill(mask == 0, 0.0)
            
            noise = torch.randint_like(action, -2, 2, dtype=torch.long)
            next_action = (self.actor_target(next_state) + noise).clamp(0, 10)
            next_action = next_action.masked_fill(mask == 0, 0)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action.squeeze())

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

#            for name, param in self.actor.named_parameters():
#                if param.requires_grad:
#                    self.mean_list.append(param.mean())
#                    self.std_list.append(param.std())
#            np.save('mean_list2.npy', self.mean_list)
#            np.save('std_list2.npy', self.std_list)


    def save(self):
        filename = self.filename
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self):
        filename = self.filename
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
