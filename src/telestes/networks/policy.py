import warnings

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from torch.distributions.categorical import Categorical

from .transformer import TransformerBlock, AttentionGate

from .utils import generate_layers

class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        transformer_layers: int = 2,
        optimizer: optim.Optimizer = optim.Adam,
        device: str = 'cpu',
        **kwargs
    ):
        gate_hparams = {
            **kwargs.get("gate", {})
        }
        network_defaults = dict(
            layers=None,
            activation_fn=None
        )
        network_architecture = {
            **network_defaults,
            **kwargs.get("network", {})
        }
        transformer_hparams = {
            **kwargs.get("transformer", {})
        }
        super(PolicyNetwork, self).__init__()

        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dims=input_dims,
                    **transformer_hparams,
                )
                for _ in range(transformer_layers)
            ]
        )
        
        self.gate = AttentionGate(
            embed_dims=input_dims,
            **gate_hparams
        )

        if network_architecture['layers'] is not None:
            layers = generate_layers(
                input_dims=input_dims,
                output_dims=output_dims,
                **network_architecture
            )
            self.actor = nn.ModuleList(layers)
        else:
            self.actor = nn.ModuleList(
                [
                    nn.Linear(input_dims, output_dims)
                ]
            )

        self.to(device)

        optimizer_hparams = {
            **kwargs.get("optimizer_hparams", {})
        }
        self.optimizer = optimizer(
            self.parameters(),
            **optimizer_hparams
        )

    def forward(self, state, mask=None):
        out = state
        for layer in self.transformer:
            out = layer(out, out, out, mask)
        out = self.gate(out)
        for layer in self.actor:
            out = layer(out)
        out = functional.softmax(out, -1)
        dist = Categorical(out)
        return dist
