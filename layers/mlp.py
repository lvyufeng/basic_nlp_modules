import torch.nn as nn
import torch

class MLP(nn.Module):
    """
    
    """    
    def __init__(self, size_of_layers, activation = 'relu', output_activation = None, dropout = 0.0):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.output_activation = output_activation
        self.output_layer = None
        for i in range(1,len(size_of_layers)):
            if i + 1 == len(size_of_layers):
                self.output_layer = nn.Linear(size_of_layers[i - 1], size_of_layers[i])
            else:
                self.hidden_layers.append(nn.Linear(size_of_layers[i - 1],size_of_layers[i]))
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_activate = [activation] * (len(size_of_layers) - 2)
    
    def forward(self, x):
        for layer, func in zip(self.hidden_layers, self.hidden_activate):
            x = self.dropout(func(layer(x)))
        x = self.output_layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        x = self.dropout(x)

        return x


