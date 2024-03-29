import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam
import sys
from torch.utils.data import DataLoader, Dataset
import math
from tqdm import tqdm
import random
import pickle
import torch.nn.functional as F
from NEURO_MOVEMENT.models.reconstruction_models import Transformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, data: list):
        self.inputs = data[0]
        self.outputs = data[1]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item], self.outputs[item]


def split_data(full_dataset, tvt_tuple):
    cut1 = math.ceil(len(full_dataset[0]) * tvt_tuple[0])
    cut2 = math.ceil(len(full_dataset[0]) * (tvt_tuple[0] + tvt_tuple[1]))
    train_data = [full_dataset[0][:cut1], full_dataset[1][:cut1]]
    valid_data = [full_dataset[0][cut1:cut2], full_dataset[1][cut1:cut2]]
    test_data = [full_dataset[0][cut2:], full_dataset[1][cut2:]]
    return train_data, valid_data, test_data


def get_batch(args, split, tvt_data):
    data = None
    if split == 'train':
        data = tvt_data[0]
    elif split == 'valid':
        data = tvt_data[1]
    elif split == 'test':
        data = tvt_data[2]
    rand = random.randint(0, len(data[0]) - args.batch_size)
    ix = [rand+j for j in range(args.batch_size)]
    print(ix)
    x = torch.stack([data[0][i] for i in ix])
    y = torch.stack([data[1][i] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for Neural Reconstruction')

    parser.add_argument('--d_model', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--n_tokens', type=int)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--tvt_tuple', type=tuple)
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('--output_dim', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lossf', choices=['L1', 'MSE', 'BCE'], type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--n_heads', choices=[4, 8, 16], type=int)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--block_size', type=int)


    parser.set_defaults(
        d_model = 256,
        dropout = 0.1,
        n_tokens = 0,
        data_dir = r"D:\Data\Research\NEURO\movement\processed_data\transformer_data.pkl",
        tvt_tuple = (0.7, 0.15, 0.15),
        input_dim = 1,
        output_dim = 1,
        batch_size = 16,
        lossf = 'L1',
        lr = 1e-4,
        n_heads = 4,
        n_layers = 3,
        block_size = 8
    )

    return parser.parse_args()


def training():
    print(f"Device: {device}")
    args = get_args()
    with open(args.data_dir, 'rb') as f:
        DATA = pickle.load(f)
    f.close()
    num_samples, args.input_dim = DATA[0].shape
    print(num_samples, args.input_dim)
    num_samples, args.output_dim = DATA[1].shape
    print(num_samples, args.output_dim)
    src_vocab, tgt_vocab = DATA[2]
    src_vocab, tgt_vocab = int(src_vocab), int(tgt_vocab)
    train_data, valid_data, test_data = split_data(DATA, args.tvt_tuple)
    tvt_data = [train_data, valid_data, test_data]
    criterion = None
    if args.lossf == 'L1':
        criterion = nn.L1Loss()
    elif args.lossf == 'MSE':
        criterion = nn.MSELoss()
    elif args.lossf == 'BCE':
        criterion = nn.BCELoss()
    d_ff = int(4 * args.d_model)
    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=args.d_model,
        num_heads=args.n_heads,
        num_layers=args.n_layers,
        d_ff=d_ff,
        max_seq_length=2000,
        dropout=args.dropout
    )
    model = model.to(device)
    x, y = get_batch(args, 'train', tvt_data)
    out = model(x, y)



if __name__ == '__main__':
    training()

