import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention

"""
from paper: 'Attention is all you need <https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>'

"""

class FFNLayer(nn.Module):
    """
    FFN(x) = max(0, x*W_1 + b_1)* W_2 + b_2
    FFN(x) = Linear(Relu(Linear(x)))
    """
    def __init__(self, model_size, inner_size, dropout = 0.1):
        super(FFNLayer, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(model_size, inner_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_size,model_size)
        )

    def forward(self, input):
        output = self.ffn(input)

        return output

class IdenticalLayer(nn.Module):
    """
    output = LayerNorm(x + MultiHeadAttention(x))
    output = LayerNorm(output + FFNLayer(output))
    """

    def __init__(self, model_size, inner_size, key_size, value_size, num_head, dropout = 0.1):
        super(IdenticalLayer,self).__init__()
        self.multi_head_attn = MultiHeadAttention(model_size, key_size, value_size, num_head, dropout)
        self.norm1 = nn.LayerNorm(model_size, eps = 1e-6)
        self.norm2 = nn.LayerNorm(model_size, eps=1e-6)
        self.ffn = FFNLayer(model_size, inner_size, dropout)

    
    def forward(self, input, seq_mask = None, attn_mask = None):
        if seq_mask == None:
            seq_mask = 1
        output = self.multi_head_attn(input, input, input, attn_mask)
        output = input + output
        output = self.norm1(output)
        output = self.ffn(output) + output
        output = self.norm2(output)

        return output
        
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **kargs):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([IdenticalLayer(**kargs) for i in range(num_layers)])

    def forward(self, x, seq_mask = None):
        if seq_mask == None:
            attn_mask = None
        else:
            pass # todo
        output = x
        for layer in self.layers:
            output = layer(output, seq_mask,attn_mask)
        
        return output
        