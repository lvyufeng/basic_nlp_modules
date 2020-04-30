import torch
import torch.nn as nn
import torch.nn.functional as F
class SelfAttention(nn.Module):
    """
    LSTM hidden state H as input,
        A = softmax(W_s2 * tanh(W_s1 * H^T))
        M = A * H
    H:      n-by-2u, u is length of hidden unit
    W_s1:   d_a-by-2u, d_a is hyper-parameter
    W_s2:   r-by-d_a, r is number of multiple hops (to focus on different parts of the sentence)
    A:      r-by-n
    M:      r-by-2u
    """
    
    def __init__(self, input_size, attention_size, attention_hops, dropout = 0.0):
        """
        input_size:     2u
        attention_size: d_a
        attention_hops: r
        """
        super(SelfAttention, self).__init__()
        self.attention_hops = attention_hops
        self.ws1 = nn.Linear(input_size, attention_size, bias= False)
        self.ws2 = nn.Linear(attention_size, attention_hops, bias= False) 
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """"
        
        """
        input = input.contiguous()
        size = input.size()

        y1 = self.tanh(self.ws1(self.dropout(input)))
        attention = self.ws2(y1).transpose(1,2).contiguous()
        
        attention = F.softmax(attention, 2)
        
        return torch.bmm(attention, input)

    def penalization_term(self):
        pass