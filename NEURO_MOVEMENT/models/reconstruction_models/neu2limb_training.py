import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot
import numpy
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from neu2limb import Transformer
from dataloaders import NeuroTransformerDataset
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='neu2limb Training Arguments')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--device', type=torch.device)
    parser.add_argument('--vocab_size1', type=int)
    parser.add_argument('--vocab_size2', type=int)
    parser.add_argument('--d_model', type=int)
    parser.add_argument('--n_head', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--n_layer', type=int)
    parser.add_argument('--bias', type=bool)
    parser.add_argument('--block_size', type=int)

    parser.set_defaults(
        data_dir = r'D:\Data\Research\NEURO\movement\processed_data/',
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        vocab_size1 = 1,
        vocab_size2 = 1,
        d_model = 256,
        n_head = 4,
        dropout = 0.1,
        n_layer = 2,
        bias = True,
        block_size = 8
    )

    return parser.parse_args()


class NoamOpt:
    def __init__(self, model_size, lr_mult, warmup, optimizer):
        self.model_size = model_size
        self.optimizer = optimizer
        self.warmup = warmup
        self.lr_mult = lr_mult
        self.steps = 0

    def step_and_update(self):
        self.update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr_scale(self):
        d_model = self.model_size
        step, warmup = self.steps, self.warmup
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** (-1.5))

    def get_cur_lr(self):
        clr = None
        for p in self.optimizer.param_groups:
            clr = p['lr']
        return clr

    def update_learning_rate(self):
        self.steps += 1
        lr = self.lr_mult * self.get_lr_scale()
        for p in self.optimizer.param_groups:
            p['lr'] = lr


def training_loop():
    args = get_args()
    with open(args.data_dir + 'general_data.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    f.close()


