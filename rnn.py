import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms


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
        # input.shape = (4, 256, 2)
        bz = inputs.shape[1] #batch_size
        ht = torch.zeros((bz,hidden_size)).to(device)
        for i in inputs:
            ht = self.rnn(i,ht)
        return ht
    
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.rnn = RNN(2,2)
        self.fc1 = nn.Linear(2,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,256)
        self.fc5 = nn.Linear(256,2)
        
    def forward(self,x):
        x = self.rnn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x