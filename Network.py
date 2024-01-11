"""
Author: Minh Pham-Dinh
Created: Jan 7th, 2024
Last Modified: Jan 10th, 2024
Email: mhpham26@colby.edu

Description:
    Network file for used with RL files
"""

import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self, inputs: int, classes: int, hidden_units: list, softmax: bool = False, std=np.sqrt(2)) -> None:
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
        if softmax:
            self.layers.append(nn.Softmax())

        def init_weights(m, std=std, bias_const=0.0):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, std)
                torch.nn.init.constant_(m.bias, bias_const)

        self.net = nn.Sequential(*self.layers)
        self.net.apply(init_weights)

    def forward(self, x: torch.Tensor):
        return self.net(x)
