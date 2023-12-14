import collections
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from memory import ReplayBuffer
from networks import Actor, Critic

#torch.cuda.set_device(2)
device = torch.device('cuda')

class AIM():
    
    def __init__(self, actor_model, actor_optimizer, critic_model_1, critic_optimizer_1,
                 critic_model_2, critic_optimizer_2, explore_noise, warmup, replay_buffer,
                 batch_size, update_interval, update_interval_actor, target_update_interval,
                 soft_update_tau, n_steps, gamma, model_name):
        
        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.critic_model_1 = critic_model_1
        self.critic_optimizer_1 = critic_optimizer_1
        self.critic_model_2 = critic_model_2
        self.critic_optimizer_2 = critic_optimizer_2
        
        self.explore_noise = explore_noise
        self.warmup = warmup
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.update_interval_actor = update_interval_actor
        self.target_update_interval = target_update_interval
        self.soft_update_tau = soft_update_tau
        self.n_steps = n_steps
        self.gamma = gamma
        self.model_name = model_name
        
        self.actor_model_target = copy.deepcopy(self.actor_model)
        self.critic_model_target_1 = copy.deepcopy(self.critic_model_1)
        self.critic_model_target_2 = copy.deepcopy(self.critic_model_2)
        
        self.time_counter = 0
        
        self.loss_record = collections.deque(maxlen=100)
        
        self.device = device
        
    def store_transition(self, nodes, edges, edges_type, action, reward, nodes_, edges_, edges_type_, done):
        
        self.replay_buffer.add(nodes, edges, edges_type, action, reward, nodes_, edges_, edges_type_, done)
        
    def sample_memory(self):
        
        samples, data_sample = self.replay_buffer.sample(self.batch_size, self.n_steps)
        
        return samples, data_sample
    
    def choose_action(self, nodes, edges, edges_type):
        
        if self.time_counter < self.warmup:
            action = np.random.normal(scale=1.7,
                                      size=(len(list(nodes.keys())),1))
            action = torch.as_tensor(action, dtype=torch.float32).to(self.device)
        else:
            action = self.actor_model.forward(nodes, edges, edges_type)
            noise = torch.as_tensor(np.random.normal(scale=self.explore_noise)).to(self.device)
            action = action + noise
   
        action = torch.clamp(action, -self.actor_model.max_action, self.actor_model.max_action)
        
        return action
    
    def test_action(self, nodes, edges, edges_type):
        
        action = self.actor_model.forward(nodes, edges, edges_type)
        
        return action
    
    def loss_process(self, loss, weight):
        
        #weight = torch.as_tensor(weight, dtype=torch.float32).to(self.device) in teoria non serve più
        #loss = torch.mean(loss*weight.detach()) in teoria non serve più
        
        return torch.mean(loss)
    
    def learn_onestep(self, info_batch, data_batch):
        def safe(el):
            return torch.as_tensor(el, dtype=torch.float32).detach()
        actor_loss = []
        critic_loss_1 = []
        critic_loss_2 = []
        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        
        # SANITY CHECK
        #tmp = [el.cpu().detach().numpy() for el in [self.critic_model_1.RGCN1.weight]]
        
        for elem in data_batch:
            nodes, edges, edges_type, action, reward, nodes_, edges_, edges_type_, done = elem
            action, reward, done = \
                [safe(el) for el in [action, reward, done]]
            action = torch.as_tensor(action, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                action_target = self.actor_model_target.forward(nodes_, edges_, edges_type_)
                action_target = action_target + \
                                torch.clamp(torch.as_tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
                action_target = torch.clamp(action_target,
                                            -self.actor_model.max_action,
                                            self.actor_model.max_action)

                q1_next = self.critic_model_target_1.forward(nodes_, edges_, edges_type_, action_target)
                q2_next = self.critic_model_target_2.forward(nodes_, edges_, edges_type_, action_target)
                critic_value_next = torch.min(q1_next, q2_next)

                critic_target = reward + self.gamma * critic_value_next * (1 - done)

            q1 = self.critic_model_1.forward(nodes, edges, edges_type, action)
            #q1 = q1.detach() in teoria non serve più
            q2 = self.critic_model_2.forward(nodes, edges, edges_type, action)
            #q2 = q2.detach() in teoria non serve più

            q1_loss = F.smooth_l1_loss(critic_target, q1)
            q2_loss = F.smooth_l1_loss(critic_target, q2)
            critic_loss_1.append(q1_loss)
            critic_loss_2.append(q2_loss)
            
        critic_loss_e_1 = torch.stack(critic_loss_1)
        critic_loss_e_2 = torch.stack(critic_loss_2)
        critic_loss_total_1 = self.loss_process(critic_loss_e_1, info_batch['weights'])
        critic_loss_total_2 = self.loss_process(critic_loss_e_2, info_batch['weights'])
        
        (critic_loss_total_1 + critic_loss_total_2).backward(retain_graph=True)
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()
        
        # SANITY CHECK
        #diff = np.mean([((t1-t2) ** 2).mean() for t1, t2 in zip(tmp, [el.cpu().detach().numpy() \
        #                            for el in [self.critic_model_1.RGCN1.weight]])])
        #print(f"diff : {diff}")
        #print(self.critic_model_1.RGCN1.weight)
    
        if self.time_counter % self.update_interval_actor != 0:
            return
        
        for elem in data_batch:
            nodes, edges, edges_type, action, reward, nodes_, edges_, edges_type_, done = elem
            
            mu = self.actor_model.forward(nodes, edges, edges_type)
            actor_loss_sample = -1 * self.critic_model_1.forward(nodes, edges, edges_type, mu)
            actor_loss_s = actor_loss_sample.mean()
            actor_loss.append(actor_loss_s)
            
        actor_loss_e = torch.stack(actor_loss)
        actor_loss_total = self.loss_process(actor_loss_e, info_batch['weights'])
        self.actor_optimizer.zero_grad()
        actor_loss_total.backward(retain_graph=True)
        self.actor_optimizer.step()
        
        self.loss_record.append(float((critic_loss_total_1 +
                                       critic_loss_total_2 +
                                       actor_loss_total).detach().cpu().numpy()))
        
    def synchronize_target(self):

        assert 0.0 < self.soft_update_tau <= 1.0

        for target_param, source_param in zip(self.critic_model_target_1.parameters(),
                                              self.critic_model_1.parameters()):
            target_param.data.copy_((1 - self.soft_update_tau) *
                                target_param.data + self.soft_update_tau * source_param.data)

        for target_param, source_param in zip(self.critic_model_target_2.parameters(),
                                              self.critic_model_2.parameters()):
            target_param.data.copy_((1 - self.soft_update_tau) *
                                target_param.data + self.soft_update_tau * source_param.data)

        for target_param, source_param in zip(self.actor_model_target.parameters(),
                                              self.actor_model.parameters()):
            target_param.data.copy_((1 - self.soft_update_tau) *
                                target_param.data + self.soft_update_tau * source_param.data)

    def learn(self):

        if self.time_counter <= self.warmup or \
            (self.time_counter % self.update_interval != 0):
            self.time_counter += 1
            return

        samples, data_sample = self.sample_memory()

        if self.n_steps == 1: # FOR THE MOMENT ALWAYS THE CASE FOR US
            self.learn_onestep(samples, data_sample)

        if self.time_counter % self.target_update_interval == 0:
            self.synchronize_target()

        self.time_counter += 1

    def get_statistics(self):

        loss_statistics = np.mean(self.loss_record) if self.loss_record else np.nan
        return [loss_statistics]

    def save_model(self, save_path):
        """
           <Model saving function>
           Used to save the trained model
        """
        save_path_actor = save_path + "/" + self.model_name + "_actor" + ".pt"
        save_path_critic_1 = save_path + "/" + self.model_name + "_critic_1" + ".pt"
        save_path_critic_2 = save_path + "/" + self.model_name + "_critic_2" + ".pt"
        torch.save(self.actor_model, save_path_actor)
        torch.save(self.critic_model_1, save_path_critic_1)
        torch.save(self.critic_model_2, save_path_critic_2)

    def load_model(self, load_path):
        """
           <model reading function>
           Used to read the trained model
        """
        load_path_actor = load_path + "/" + self.model_name + "_actor" + ".pt"
        load_path_critic_1 = load_path + "/" + self.model_name + "_critic_1" + ".pt"
        load_path_critic_2 = load_path + "/" + self.model_name + "_critic_2" + ".pt"
        self.actor_model = torch.load(load_path_actor)
        self.critic_model_1 = torch.load(load_path_critic_1)
        self.critic_model_2 = torch.load(load_path_critic_2)
