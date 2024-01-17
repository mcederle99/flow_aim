import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import Actor, Critic

class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.3,
        policy_freq=2,
        filename='LSTM_AIM'
    ):

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=1e-6)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.filename = filename

        self.total_it = 0


    def select_action(self, state):
        return self.actor.forward(state).detach().cpu()
    
    def mse(self, expected, targets, weights):
        """Custom loss function that takes into account the importance-sampling weights."""
        td_error = expected - targets
        weighted_squared_error = weights * td_error * td_error
        return torch.sum(weighted_squared_error) / torch.numel(weighted_squared_error)

    def train(self, replay_buffer, batch_size=128):
        self.total_it += 1
        
        for i in range(1, 11):
            # Sample replay buffer 
            batch = replay_buffer.sample(batch_size)
            state, action, next_state, reward, not_done = batch['obs'], batch['action'], batch['next_obs'], batch['reward'], batch['not_done']
            action = torch.tensor(action, device=self.actor.device, dtype=torch.float32)
            reward = torch.tensor(reward, device=self.actor.device, dtype=torch.float32)
            not_done = torch.tensor(not_done, device=self.actor.device)

            weights = torch.tensor(batch['weights'], device=self.actor.device, dtype=torch.float32)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                next_action = (
                    self.actor_target.forward_train(next_state) + noise
                ).clamp(-1, 1)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = self.mse(current_Q1, target_Q, weights) + self.mse(current_Q2, target_Q, weights)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()

            errors1 = np.abs((current_Q1 - target_Q).detach().cpu().numpy())
            replay_buffer.update_priorities(batch['indexes'], errors1)

            # Delayed policy updates
            if i % self.policy_freq == 0:

                # Compute actor losse
                actor_loss = -self.critic.Q1(state, self.actor.forward_train(state)).mean()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


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
