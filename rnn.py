import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
hidden_size = 2

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNNCell(input_dim,hidden_size)

    def forward(self, inputs):
        # input.shape = (?, batch_size, 5)
        bz = inputs.shape[1] #batch_size
        ht = torch.zeros((bz,hidden_size)).to(device)
        for i in inputs:
            ht = self.rnn(i,ht)
        return ht
    
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        num_info = 3
        self.lstm = nn.LSTM(num_info, 32, proj_size=num_info, batch_first=True)
        
    def forward(self, X): # X: [batch_size, seq_length, 3]
        
        _, (h_n_1, _) = self.lstm(X)
        h_n_1 = F.sigmoid(h_n_1)
        
        # h_n_1: [30~100, 0~1, 0.01~1]
        h_n_1[0] = 30 + 70*h_n_1[0]
        h_n_1[2] = 0.01 + 0.99*h_n_1[2]
        
        return h_n_1
    
    def save_model(self, net, filename):
        torch.save(net.state_dict(), filename)
