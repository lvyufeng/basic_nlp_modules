import torch
import torch.nn as nn
import torch.nn.functional as F
# default torch version
# nn.LSTM(input_size, hidden_size, layer_num)

# use Linear layers
class simpleLSTM(nn.Module):
    '''
    forget_gate:        f_t = sigmoid(W_f[h_(t-1),x_t] + b_f)
    input_gate:         i_t = sigmoid(W_i[h_(t-1),x_t] + b_i)
    output_gate:        o_t = sigmoid(W_o[h_(t-1),x_t] + b_o)
    cell_state:         C_t = tanh(W_C[h_(t-1),x_t] + b_C)
    pre_cell_state:     C_(t-1)
    updated_cell_state: C_t^u = f_t * C_(t-1) + i_t * C_t
    hidden_state:       h_t = o_t * tanh(C_t^u)  
    '''
    def __init__(self, input_size, hidden_size, layer_num = None):
        super(simpleLSTM,self).__init__()
        # self.cell_size = cell_size
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.forget_gate = nn.Linear(hidden_size + input_size, hidden_size) # []
        self.input_gate = nn.Linear(hidden_size + input_size, hidden_size)
        self.output_gate = nn.Linear(hidden_size + input_size, hidden_size)
        self.cell_state = nn.Linear(hidden_size + input_size, hidden_size)

    def _hidden_init(self, batch_size):
        '''
        input: batch_size
        output: [batch_size, hidden_size], [batch_size, hidden_size]
        '''
        
        return torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
    def _forward(self, x, pre_hidden_state, pre_cell_state):
        '''
        map the formula
        '''
        value = torch.cat((x, pre_hidden_state), 1) # [h_(t-1),x]
        forget_value = F.sigmoid(self.forget_gate(value)) # forget_gate
        input_value = F.sigmoid(self.input_gate(value)) # input gate
        output_value = F.sigmoid(self.output_gate(value)) # output gate
        cell_state = F.tanh(self.cell_state(value)) # cell_state
        updated_cell_state = forget_value * pre_cell_state + input_value * cell_state
        hidden_state = output_value * F.tanh(updated_cell_state)

        return hidden_state, updated_cell_state
    
    def forward(self, inputs):
        '''
        inputs: [batch_size, length, embedding_size]
        '''
        batch_size = inputs.size(0)
        time_step = inputs.size(1)

        hidden_state, cell_state = self._hidden_init(batch_size)

        for i in range(time_step):
            hidden_state, cell_state = self._forward(torch.sequeeze(inputs[:,i:i+1,:]), hidden_state, cell_state)
        return hidden_state, cell_state
