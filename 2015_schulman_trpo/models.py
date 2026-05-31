import torch
import torch.nn as nn


class GaussPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        m = self.action_mean(x)
        ls = self.action_log_std.expand_as(m)
        return m, ls, torch.exp(ls)


class CategoricalPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.head = nn.Linear(64, num_outputs)
        self.head.weight.data.mul_(0.1)
        self.head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.head(x)


class Value(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.head = nn.Linear(64, 1)
        self.head.weight.data.mul_(0.1)
        self.head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.head(x)
