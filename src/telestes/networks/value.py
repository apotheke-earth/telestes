import warnings

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from .transformer import TransformerBlock, AttentionGate

from .utils import generate_layers

class ValueNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        transformer_layers: int = 2,
        optimizer: optim.Optimizer = optim.Adam,
        device='cpu',
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
        super(ValueNetwork, self).__init__()

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
                output_dims=1,
                **network_architecture
            )
            self.critic = nn.ModuleList(layers)
        else:
            self.critic = nn.ModuleList(
                [
                    nn.Linear(input_dims, 1)
                ]
            )

        optimizer_hparams = {
            **kwargs.get("optimizer_hparams", {})
        }
        self.optimizer = optimizer(
            self.parameters(),
            **optimizer_hparams
        )

        self.to(device)

    def forward(self, state, mask=None):
        out = state
        for layer in self.transformer:
            out = layer(out, out, out, mask)
        out = self.gate(out)
        for layer in self.critic:
            out = layer(out)
        value = out
        return value
