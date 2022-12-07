import random
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import pretty_midi
import pathlib
import glob
import os
import pickle
from rnn import Net

from rl_network import actor,critic
from replaybuffer import Memory

# from train_RL import sample as SAMPLE
CHECKPOINT= os.path.join(os.path.dirname(__file__), 'models/LSTM')
CHECKPOINT_CRITIC = os.path.join(os.path.dirname(__file__), 'models/critic')
CHECKPOINT_ACTOR = os.path.join(os.path.dirname(__file__), 'models/actor')

class ddpg:
    def __init__(self,n_split = 4,is_sample = False,sample = 0):
        with open(os.path.dirname(os.path.realpath(__file__))+'/reward_data/reward.pickle', 'wb') as f:
            pass
        ########hyperparameter########
        self.lr = 0.001
        self.memory_capacity = 100000
        self.max_episode = 10000
        self.n_split = n_split
        self.epi_step = 500
        self.batch_size = 2
        self.polyak = 0.95 #soft update target network
        self.gamma = 0.99
        
        # is sample
        self.is_sample = is_sample
        
        # cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # loading LSTM model
        self.net = Net().to(self.device)
        self.net.load_state_dict(torch.load(CHECKPOINT + '.pt', map_location=torch.device(self.device)))
        
        # loading sample
        if is_sample:
            self.sample = sample
        else:
            self.sample = self.loading_sample(n_split = self.n_split)
            
        print(self.sample)
        
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
        
    def loading_sample(self, n_split = 4): #how many past note is considered?
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
                step = note.start-prev_start
                duration = note.end - note.start
                total_time_interval += duration
                total_note += note.pitch
                sample.append([note.pitch,step,duration])
                prev_start = note.start
            train_samples.append(sample)
            j += 1
            # break
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
    
    def reset(self,samples):
        mini_batch = random.sample(samples,1)
        initial_state = mini_batch[0][0]
        initial_state = torch.Tensor(np.array(initial_state)).to(self.device)
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
        actor_loss = -self.critic_network(states,actions_real).mean()
        
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
        action = action.cpu().detach().numpy()
        action += np.array([7,0.1,0.099])
        action = np.clip(action,np.array([30,0,0.01]),np.array([100,1,1]))
        random_actions = np.array([random.uniform(30,100),random.uniform(0,1),random.uniform(0.01,1)])
        action += np.random.binomial(1, 0.3, 3)[0] * (random_actions - action)
        return action
    
    def get_reward(self,state,action):
        state = state.cpu().detach().numpy()
        normalized_error = (state - action)/np.array([70,1,0.99])
        reward = -(normalized_error**2).mean()
        return reward
    
    def learn(self):
        for episode in range(1,self.max_episode+1):
            s = self.reset(self.sample)
            done = False
            cumulative_reward = 0
            for _ in range(self.epi_step):
                state = self.net(s)
                if not(_ == 0):
                    action = action.squeeze()
                    prev_state_ = prev_state.cpu().detach().numpy()
                    state_ = state.cpu().detach().numpy()
                    action_ = action.cpu().detach().numpy()
                    self.buffer.push(prev_state_,state_,action_,reward,done)
                action = self.select_action(self.actor_network(state))
                #학습된 lstm의 output은 state지만, actor network의 output은 action이다. \
                    # 그 차이를 좁히기 위해 거리에 마이너스를 취해 리워드로 만든다.
                reward = self.get_reward(state,action)
                action = torch.Tensor([action]).to(self.device)
                s = torch.cat([s[-(self.n_split-1):],action],dim=0)
                prev_state = state
                cumulative_reward += reward
                if _ == self.epi_step-2:
                    done = True
                if len(self.buffer) > self.batch_size:
                    self.update_network()
                    self.soft_update_target_network(self.target_actor_network,self.actor_network)
                    self.soft_update_target_network(self.target_critic_network,self.critic_network)
            self.actor_network.save_model(self.actor_network,CHECKPOINT_CRITIC+'.pt')
            self.critic_network.save_model(self.critic_network,CHECKPOINT_ACTOR+'.pt')
            print("============================================")
            print('(',episode,'/',self.max_episode,')','episodes done')
            print("cumulative reward: ",cumulative_reward)
            print("============================================")
            with open(os.path.dirname(os.path.realpath(__file__))+'/reward_data/reward.pickle', 'ab') as f:
                pickle.dump(cumulative_reward,f)