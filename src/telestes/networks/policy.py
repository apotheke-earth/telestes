import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dists

from .architectures.transformer import TransformerBlock, AttentionGate
from .architectures.linear import LinearBlock


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        transformer_layers: int = 2,
        linear_layers: int = 2,
        optimizer_class: optim.Optimizer = optim.Adam,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Initialize the Policy Network (Actor in A/C settings).
        """
        super(PolicyNetwork, self).__init__()

        transformer_hparams = {
            **kwargs.get("transformer", {})
        }
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dims=input_dims,
                    **transformer_hparams
                )
                for _ in range(transformer_layers)
            ]
        )

        gate_hparams = {
            **kwargs.get("gate", {})
        }
        self.gate = AttentionGate(
            embed_dims=input_dims,
            **gate_hparams
        )

        linear_hparams = {
            **kwargs.get("linear", {})
        }
        self.linear = nn.Sequential(
            LinearBlock(
                input_dims=input_dims,
                output_dims=output_dims,
                **linear_hparams
            ),
            nn.Softmax(dim=-1)
        )
        print(self.linear)

        self.to(device)

        optimizer_hparams = {
            **kwargs.get("optimizer", {})
        }
        self.optimizer = optimizer_class(
            self.parameters(),
            **optimizer_hparams
        )

    def forward(
        self,
        state: torch.Tensor,
        mask=None
    ) -> dists.distribution.Distribution:
        """
        Output a distribution over the action space based on the current state.
        """
        out = state
        for layer in self.transformer:
            out = layer(out, out, out, mask)
        out = self.gate(out)
        out = self.linear(out)
        dist = dists.categorical.Categorical(out)
        return dist
        

