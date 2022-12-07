import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define the actor network
class actor(nn.Module):
    def __init__(self):
        super(actor, self).__init__()
        self.max_pitch = 70
        self.max_step = 1
        self.max_duration = 0.99
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = F.sigmoid(self.action_out(x))
        return actions
    
    def save_model(self, net, filename):
        torch.save(net.state_dict(), filename)

class critic(nn.Module):
    def __init__(self):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(3+3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value
    
    def save_model(self, net, filename):
        torch.save(net.state_dict(), filename)