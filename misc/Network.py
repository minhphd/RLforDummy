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

def init_weights(m, std=np.sqrt(2), bias_const=0.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.orthogonal_(m.weight, std)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, bias_const)

class Net(nn.Module):
    def __init__(self, inputs: int, classes: int, hidden_units: list, softmax: bool = False, open_ended=False) -> None:
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
            if open_ended:
                raise Exception('cannot create shared network with softmax layer, please set softmax=False')
            self.layers.append(nn.Softmax())
        elif open_ended:
            self.layers.append(nn.ReLU())    

        self.net = nn.Sequential(*self.layers)
        self.net.apply(init_weights)

    def forward(self, x: torch.Tensor):
        return self.net(x)
    

class CNNnetwork(nn.Module):
    def __init__(self, input_shape, output_class, hidden_units, softmax=False, open_ended=False):
        super(CNNnetwork, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.conv_layers.apply(init_weights)

        # Compute the output shape after the convolutional layers
        with torch.no_grad():
            self.feature_size = self._get_conv_output(input_shape)

        # dense layers
        self.dense_layers = Net(self.feature_size, output_class, hidden_units, softmax, open_ended)
        
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.conv_layers(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x