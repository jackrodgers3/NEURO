import random
import sys
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Cortex
import argparse
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset

parser = argparse.ArgumentParser(description='Cortex PreTraining Arguments')
parser.add_argument('--src_vocab_size', help='n/a', type=int)
parser.add_argument('--tgt_vocab_size', help='n/a', type=int)
parser.add_argument('--d_model', help='model embedding dimension', type=int)
parser.add_argument('--num_heads', help='number of attention heads', type=int)
parser.add_argument('--num_layers', help='number of layers for enc/dec', type=int)
parser.add_argument('--max_seq_length', help='sequence length', type=int)
parser.add_argument('--dropout', help='dropout rate', type=float)
parser.add_argument('--lr', help='learning rate', type=float)
parser.add_argument('--loss_function', help='loss function', type=int)
parser.add_argument('--label_smoothing', help='label smoothing', type=float)
parser.add_argument('--batch_size', help='batch size', type=int)
parser.add_argument('--rep', help='reproducibility', type=bool)
parser.add_argument('--mi', help='maximum iterations', type=int)

parser.set_defaults(src_vocab_size = 1719, tgt_vocab_size = 1719, d_model = 128,
                    num_heads = 8, num_layers = 3, max_seq_length = 32, dropout=0.1,
                    lr=3e-4, loss_function=3, label_smoothing=0.1,
                    batch_size=10, rep=True, mi=1000)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomNeuralDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SeminalOpt:
    def __init__(self, model_size, lr_mult, optimizer, warmup):
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
        return (d_model ** -0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

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


def split(raw_data):
    n = int(0.95* len(raw_data))
    train_data = raw_data[:n]
    val_data = raw_data[n:]
    return train_data, val_data


def get_batch(raw_data, args):
    max_len, vocab_size = raw_data.shape[0] - 1, raw_data.shape[1]
    b = random.randint(0, max_len - args.batch_size)
    xb = torch.empty(size = (args.batch_size, vocab_size), dtype=torch.float, device=device)
    yb = torch.empty(size = (args.batch_size, vocab_size), dtype=torch.float, device=device)
    for i in range(args.batch_size):
        xb[i] = raw_data[b + i]
        yb[i] = raw_data[b + i + 1]
    return xb, yb


def train(args):
    if args.rep:
        torch.manual_seed(23)
    with open(r'C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023\Transformer\inputs_pickle', 'rb') as f:
        raw_data = pickle.load(f)
    f.close()
    num_neurons = 1719
    args.src_vocab_size = num_neurons
    args.tgt_vocab_size = num_neurons
    model = Cortex(args.src_vocab_size, args.tgt_vocab_size, args.d_model, args.num_heads, args.num_layers, 4*args.d_model, args.max_seq_length, args.dropout)
    model = model.to(device)
    criterion = None
    if args.loss_function == 1:
        criterion = nn.L1Loss(reduction='mean')
    elif args.loss_function == 2:
        criterion = nn.MSELoss(reduction='mean')
    elif args.loss_function == 3:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    wandb.init(
        project='NEURO_TRANSFORMER_RECONSTRUCTION',
        config={
            'block_size': args.batch_size,
            'num_layers': args.num_layers,
            'max_iters': args.mi,
            'label_smoothing': args.label_smoothing,
            'd_model': args.d_model,
            'dropout': args.dropout
        }
    )
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    sched = SeminalOpt(args.d_model, 1.0, optimizer, warmup=5000)
    wandb.watch(model, criterion=F.cross_entropy, log='all', log_freq=10, log_graph=True)
    model.train()
    for iter in tqdm(range(args.mi)):
        sched.zero_grad()
        xb, yb = get_batch(raw_data, args)
        xb = xb.type(torch.LongTensor)
        yb = yb.type(torch.LongTensor)
        output = model(xb, yb[:, :-1])

        loss = criterion(output.contiguous().view(-1, args.tgt_vocab_size), yb[:, 1:].contiguous().view(-1))
        wandb.log({'loss': loss.item(), 'cur_lr': sched.get_cur_lr()})
        if iter % 100 == 0 and iter != 0:
            print(loss.item())
        loss.backward()
        sched.step_and_update()

    wandb.finish()


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)

