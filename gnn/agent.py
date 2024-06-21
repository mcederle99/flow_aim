import copy
import torch
import torch.nn.functional as f
from torch_geometric.loader import DataLoader
from nnetworks import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3(object):
    def __init__(
            self,
            state_dim,
            edge_dim,
            action_dim,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            max_action=1
    ):

        self.actor = Actor(state_dim, edge_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, edge_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_action = max_action

        self.total_it = 0

    def select_action(self, state, edge_index, edge_attr):
        return self.actor(state, edge_index, edge_attr).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        b = replay_buffer.sample(batch_size)
        loader = DataLoader(b, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
        batch = next(iter(loader))

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_action = self.actor_target(batch.x_t, batch.edge_index_t, batch.edge_attr_t)
            noise = (
                    torch.randn_like(next_action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target q value
            target_q1, target_q2 = self.critic_target(batch, next_action, 't')
            target_q = torch.min(target_q1, target_q2)
            target_q = batch.reward.unsqueeze(dim=1) + batch.not_done.unsqueeze(dim=1) * self.discount * target_q
        # Get current q estimates
        current_q1, current_q2 = self.critic(batch, batch.actions, 's')
        # Compute critic loss
        critic_loss = f.mse_loss(current_q1, target_q) + f.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losses
            actor_loss = -self.critic.q1(batch, self.actor(batch.x_s, batch.edge_index_s, batch.edge_attr_s), 's').mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
