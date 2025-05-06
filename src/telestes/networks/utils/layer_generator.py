import torch.nn as nn

def generate_layers(
    input_dims: int,
    output_dims: int,
    layers: list,
    activation_fn: str = 'none'
):

    activation_fn = activation_fn.lower()


    generated_layers = [
        nn.Linear(input_dims, layers[0]),
    ]

    for i in range(len(layers)-1):
        if activation_fn == 'relu':
            generated_layers.append(
                nn.ReLU()
            )
        elif activation_fn == 'leakyrelu':
            generated_layers.append(
                nn.LeakyReLU()
            )
        elif activation_fn == 'tanh':
            generated_layers.append(
                nn.Tanh()
            )

        generated_layers.append(
            nn.Linear(layers[i], layers[i+1])
        )

    return generated_layers

