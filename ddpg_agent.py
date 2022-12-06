import random
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import pretty_midi
import pathlib
import glob

from rl_network import actor,critic
from replaybuffer import Memory


class ddpg:
    def __init__(self):
        ########hyperparameter########
        self.lr = 0.001
        self.memory_capacity = 100000
        self.max_episode = 100000
        self.n_split = 4
        self.epi_step = 100
        self.batch_size = 256
        self.polyak = 0.95 #soft update target network
        self.gamma = 0.99
        
        # loading sample
        self.sample = self.loading_sample(n_split = self.n_split)
        
        #cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # create the network
        self.actor_network = actor().to(self.device)
        self.critic_network = critic().to(self.device)
        
        # build up the target network
        self.target_actor_network = actor().to(self.device)
        self.target_critic_network = critic().to(self.device)
        
        # load the weights into the target networks
        self.target_actor_network.load_state_dict(self.actor_network.state_dict())
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())
        
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.lr)
        
        # create the replay buffer
        self.buffer = Memory(self.memory_capacity)
        
    def loading_sample(n_split = 4): #how many past note is considered?
        total_time_interval = 0
        total_note = 0
        data_dir = pathlib.Path('data/maestro-v2.0.0')
        if not data_dir.exists():
            tf.keras.utils.get_file(
                'maestro-v2.0.0-midi.zip',
                origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
                extract=True,
                cache_dir='.', cache_subdir='data',
            )
        filenames = glob.glob(str(data_dir/'**/*.mid*'))
        train_samples = []
        j = 1
        for filename in filenames:
            print("loading sample:",str(j)+'/'+str(len(filenames)))
            sample_file = filename
            pm = pretty_midi.PrettyMIDI(sample_file)
            instrument = pm.instruments[0]
            sample = []
            sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
            prev_start = sorted_notes[0].start
            for i, note in enumerate(sorted_notes):
                start = note.start
                end = note.end
                step = note.start-prev_start
                duration = note.end - note.start
                total_time_interval += duration
                total_note += note.pitch
                sample.append([note.pitch,start,end,step,duration])
                prev_start = note.start
            train_samples.append(sample)
            j += 1
        splited_part = []
        unit_number = n_split
        j = 1
        for sample in train_samples:
            i = 0
            print("spliting sample:",str(j)+'/'+str(len(filenames)))
            while True:
                if len(sample)-(unit_number) == i:
                    break
                splited_part.append([sample[i:i+unit_number],sample[i+unit_number]])
                i += 1
            j += 1
        random.shuffle(splited_part)
        print(len(splited_part),"split melodies are shuffled!")
        print('================================================')
        return splited_part
    
    def reset(samples):
        mini_batch = random.sample(samples,1)
        initial_state = mini_batch[0][0]
        return initial_state
    
    def update_network(self):
        batch = self.buffer.sample(self.batch_size)
        states = torch.Tensor(batch.state).to(self.device)
        next_states = torch.Tensor(batch.next_state).to(self.device)
        actions = torch.Tensor(batch.action).long().to(self.device)
        rewards = torch.Tensor(batch.reward).to(self.device)
        dones = torch.Tensor(batch.done).to(self.device)
        ####################################################### normalization 한번 해야하지 않을까?
        with torch.no_grad():
            action_next = self.target_actor_network(next_states)
            q_next_values = self.target_critic_network(next_states,action_next)
            q_next_values = q_next_values.detach()
            target_q_value = rewards + self.gamma*q_next_values*(1-dones)
            target_q_value = target_q_value.detach()
        q_values = self.critic_network(states,actions)
        critic_loss = (target_q_value-q_values).pow(2).mean()
        actions_real = self.actor_network(states)
        actor_loss= -self.critic_network(states,actions_real).mean()
        
        #update
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
    
    def soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.polyak)*param.data + self.polyak*target_param.data)
        
    def select_action(self,action): # to exploration
        action = action.cpu().numpy().squeeze()
        action += np.array([7,0.1,0.099])
        action = np.clip(action,np.array([30,0,0.01]),np.array([100,1,1]))
        random_actions = np.array([random.uniform(30,100),random.uniform(0,1),random.uniform(0.01,1)])
        action += np.random.binomial(1, 0.3, 3)[0] * (random_actions - action)
        return action
    
    def learn(self):
        for episode in range(1,self.max_episode+1):
            s = self.reset(self.sample)
            done = False
            for _ in range(self.epi_step):
                #state = pretrained_lstm_model(s)
                #store_memory(prev_state,state,action,reward,done) #첫 스텝은 건너뛰도록 if문 설정하기
                #action = select_action(actor(state))
                #reward = -norm(dist(state,action)) #학습된 lstm의 output은 state지만, actor network의 output은 action이다. 그 차이를 좁히기 위해 거리에 마이너스를 취해 리워드로 만든다.
                #s = [s[:-(n_split)],action]
                #prev_state = state
                if _ == self.epi_step-2:
                    done = True
                if len(self.buffer) > self.batch_size:
                    self.update_network()
                    self.soft_update_target_network(self.target_actor_network,self.actor_network)
                    self.soft_update_target_network(self.target_critic_network,self.critic_network)
                    