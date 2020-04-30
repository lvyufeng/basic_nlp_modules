import torch
from .scaled_dot_product_attention import ScaledDotAttention
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    from paper: 'Attention is all you need <https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>'

    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^V)
    """
    def __init__(self, input_size, key_size, value_size, num_head, dropout = 0.0):
        """
        
        """
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_head = num_head

        in_size = key_size * num_head # input_size of W^Q, W^K, W^V
        self.q = nn.Linear(in_size, input_size)
        self.k = nn.Linear(in_size,input_size)
        self.v = nn.Linear(in_size,input_size)
        self.attention = ScaledDotAttention(key_size,value_size,dropout=dropout)
        self.out = nn.Linear(value_size * num_head, input_size) # W^O

    def forward(self, Q, K, V, attn_mask = None):
        batch_size = Q.size(0)
        seq_len_q = Q.size(1)
        seq_len_k = K.size(1)
        q = self.q(Q).view(batch_size, seq_len_q, self.num_head, self.key_size).transpose(1,2) #[batch, n_head, seq_len_q, key_size]
        k = self.k(K).view(batch_size, seq_len_k, self.num_head, self.key_size).transpose(1,2) #[batch, n_head, seq_len_k, key_size]
        v = self.v(V).view(batch_size, seq_len_k, self.num_head, self.value_size).transpose(1,2) #[batch, n_head, seq_len_k, value_size]

        attn = self.attention(q, k, v, attn_mask).view(batch_size, self.num_head, seq_len_q, self.value_size) #[batch, n_head, seq_len_q, value_size]
        # concat all heads
        attn = attn.transpose(1,2).contiguous.view(batch_size,seq_len_q,-1) #[batch, seq_len_q, n_head * value_size]
        output = self.out(attn) #[batch, seq_len_q, input_size]
        return output
