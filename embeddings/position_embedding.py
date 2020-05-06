import torch
import numpy as np

def position_encoding(n_postion, dim_hidden, padding_idx = None):
    """
    from paper: Attention is all you need <http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>
    """
    def angle(postion, hidden_idx):
        """
        pos/10000^(2i/d_{model})
        """
        return postion / np.power(10000, 2 * (hidden_idx // 2) / dim_hidden)
    def position_angle_vec(postion):
        return [angle(postion, hidden_idx) for hidden_idx in range(dim_hidden)]

    sin_table = np.array([position_angle_vec(pos) for pos in range(n_postion)])
    sin_table[:, 0::2] = np.sin(sin_table[:, 0::2])
    sin_table[:, 1::2] = np.cos(sin_table[:, 1::2]) 

    if padding_idx is not None:
        sin_table[padding_idx] = 0.

    return torch.FloatTensor(sin_table)   


import torch.nn as nn
class PostionEmbedding(nn.Module):
    def __init__(self):
        super(PostionEmbedding, self).__init__()

    def forward(self,):
        pass