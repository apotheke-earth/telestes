import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        layers: list[int] = None,
        activation_fn: nn.Module = nn.ReLU
    ):
        super(LinearBlock, self).__init__()

        layers = layers if layers is not None else []
        layer_dims = [input_dims] + layers + [output_dims]
        network = []

        for i in range(len(layer_dims)-1):
            network.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                network.append(activation_fn())
            
        self.network = nn.Sequential(*network)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
