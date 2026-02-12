import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear2DBias(nn.Linear):
    """Linear layer with bias stored as 2D parameter (1, out_features) for strict Muon.
    Subclasses nn.Linear so PyTorch's quantize_dynamic can quantize it."""

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        # Store bias as (1, out_features) for Muon compatibility; nn.Linear uses (out_features,)
        self.bias = nn.Parameter(self.bias.unsqueeze(0))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias.reshape(-1))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=4, activation=F.relu, output_dim=3, sigmoid_output=False):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.depth = depth
        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output

        self.layers = nn.ModuleList([Linear2DBias(input_dim, hidden_dim)])
        for _ in range(depth - 2):
            self.layers.append(Linear2DBias(hidden_dim, hidden_dim))
        self.layers.append(Linear2DBias(hidden_dim, output_dim))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        
        return x
