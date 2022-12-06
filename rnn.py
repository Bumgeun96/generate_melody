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
        # input.shape = (?, batch_size, 5)
        bz = inputs.shape[1] #batch_size
        ht = torch.zeros((bz,hidden_size)).to(device)
        for i in inputs:
            ht = self.rnn(i,ht)
        return ht
    
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.lstm = nn.LSTM(3,32, batch_first=True, bidirectional=True)
        
    def forward(self, X, x5): # X: [batch_size, seq_length, 5], x5: [batch_size, 5]
        _, (h_n_1, _) = self.lstm(X)
        score = torch.bmm(h_n_1, x5)
        loss = -F.logsigmoid(score).squeeze()
        return loss.mean()
