import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import  Module
import torch.nn.functional as F

class MLPEvaluation(Module):
    def __init__(self, emb_size):
        super(MLPEvaluation, self).__init__()

        self.W0 = Parameter(torch.FloatTensor(emb_size, emb_size), requires_grad=True)
        self.W1 = Parameter(torch.FloatTensor(emb_size, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.W0)
        nn.init.xavier_uniform_(self.W1)


    def forward(self, inputs):
        x = F.relu(torch.matmul(inputs, self.W0))
        x = torch.sigmoid(torch.matmul(x, self.W1))
        return x