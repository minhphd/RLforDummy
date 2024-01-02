import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, inputs: int, classes: int, hidden_units: list) -> None:
        super(Net, self).__init__()
        self.inputs = inputs
        self.classes = classes
        self.layers = [
            nn.Linear(inputs, hidden_units[0]),
            nn.ReLU()
        ]

        # iteratively add hidden layers
        for i in range(1, len(hidden_units), 1):
            self.layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            self.layers.append(nn.ReLU())

        # add output layer
        self.layers.append(nn.Linear(hidden_units[-1], classes))
        self.layers.append(nn.Identity())

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.net = nn.Sequential(*self.layers)
        self.net.apply(init_weights)

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x[None, :]
        mean = torch.mean(x)
        std = torch.std(x)

        x_standardized = (x - mean) / std
        return self.net(x_standardized)
