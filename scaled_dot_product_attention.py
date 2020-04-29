import torch
import torch.nn as nn
import math

class ScaledDotAttention(nn.Module):
    """
    Attention(Q,K,V) = softmax(Q*K^T/sqrt(d_k))*V
    """
    def __init__(self, key_size, value_size, dropout = 0.0):
        super(ScaledDotAttention,self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, Q, K, V, mask = None):
        
        output = torch.matmul(Q,K.transpose(-1,-2)) / self.scale
        if mask:
            output.masked_fill_(mask, -1e9)
        output = self.softmax(output)
        output = self.dropout(output)
        return torch.matmul(output,V)