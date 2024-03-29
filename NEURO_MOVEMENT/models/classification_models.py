import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, hidden_dim, act_fn, dropout, bias, residual):
        super(Block, self).__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual
        if act_fn == 'relu':
            self.act = nn.ReLU()
        elif act_fn == 'leakyrelu':
            self.act = nn.LeakyReLU()
        self.dropout = dropout
        self.bias = bias
        self.main = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias = self.bias),
            nn.BatchNorm1d(self.hidden_dim),
            self.act,
            nn.Dropout(p = dropout)
        )

    def forward(self, x):
        if self.residual:
            out = x + self.main(x)
        else:
            out = self.main(x)
        return out


class DNN(nn.Module):
    def __init__(self, linear_stack):
        super(DNN, self).__init__()
        self.linear_stack = linear_stack

    def forward(self, x):
        predictions = self.linear_stack(x)
        return predictions


def define_dnn(
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        act_fn: str,
        dropout: float,
        bias: bool,
        residual: bool,
        task: str
):
    assert act_fn == 'relu' or act_fn == 'leakyrelu'
    act = None
    if act_fn == 'relu':
        act = nn.ReLU()
    elif act_fn == 'leakyrelu':
        act = nn.LeakyReLU()

    layers = [nn.Linear(input_dim, hidden_dim, bias = bias), nn.BatchNorm1d(hidden_dim), act]

    for _ in range(num_layers - 1):
        layers.append(Block(hidden_dim, act_fn, dropout, bias, residual))

    layers.append(nn.Linear(hidden_dim, output_dim, bias = bias))

    if task == 'multiclass':
        layers.append(nn.Softmax())
    elif task == 'binaryclass':
        layers.append(nn.Sigmoid())
    elif task == 'reconstruction':
        pass

    stack = nn.Sequential(*layers)

    model = DNN(stack)

    return model

