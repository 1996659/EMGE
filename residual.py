import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
class BiLstm_reason(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLstm_reason, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.bn = nn.BatchNorm1d(2*hidden_size)

    def forward(self, x):
        output, (h_n, c_n) = self.bilstm(x)
        output = self.bn(output.transpose(1, 2)).transpose(1, 2)
        return output

  