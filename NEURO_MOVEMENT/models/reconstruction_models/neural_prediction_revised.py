import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pickle
import sys
import wandb
from tqdm import tqdm


class Head(nn.Module):
    def __init__(self, d_model, head_size, bias, dropout, block_size):
        super(Head, self).__init__()
        self.key = nn.Linear(d_model, head_size, bias)
        self.query = nn.Linear(d_model, head_size, bias)
        self.value = nn.Linear(d_model, head_size, bias)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, head_size, bias, dropout, block_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(d_model, head_size, bias, dropout, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, bias, block_size):
        super(Block, self).__init__()
        d_k = d_model // n_head
        self.ffwd = FeedForward(d_model, dropout)
        self.sa = MultiHeadAttention(d_model, n_head, d_k, bias, dropout, block_size)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, dropout, n_layer, bias, block_size, device):
        super(Transformer, self).__init__()
        self.token_embedding = nn.Linear(vocab_size, d_model, bias)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_head, dropout, bias, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.block_size = block_size
        self.device = device
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets = None):
        B, T, C = idx.shape
        # print(B, T)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        #idx is (B, T)
        for _ in range(max_new_tokens):
            # crop idx to last tokens
            idx_cond = idx[:, -self.block_size:, :]
            # get predictions
            logits = self(idx_cond)
            # focus on last time token
            logits = logits[:, -1, :]  # (B, C)
            logits = torch.unsqueeze(logits, dim=0)
            # apply softmax for probabilities
            # probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            # idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to running sequence
            idx = torch.cat((idx, logits), dim=1)
        return idx


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


def split_data(full_data, tvt_tuple):
    n1 = int(tvt_tuple[0] * len(full_data))
    n2 = int((tvt_tuple[0] + tvt_tuple[1]) * len(full_data))
    train_data = full_data[:n1]
    valid_data = full_data[n1:n2]
    test_data = full_data[n2:]
    return [train_data, valid_data, test_data]


def get_batch(split, tvt_data, args):
    data = None
    if split == 'train':
        data = tvt_data[0]
    elif split == 'valid':
        data = tvt_data[1]
    elif split == 'test':
        data = tvt_data[2]
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([data[i:i+args.block_size] for i in ix]).type(torch.float)
    y = torch.stack([data[i+1:i+args.block_size+1] for i in ix]).type(torch.float)
    return x, y


def get_gen_batch(split, tvt_data, args):
    data = None
    if split == 'train':
        data = tvt_data[0]
    elif split == 'valid':
        data = tvt_data[1]
    elif split == 'test':
        data = tvt_data[2]
    ix = torch.randint(len(data) - args.block_size, (1,))
    x = torch.stack([data[i:i+args.block_size] for i in ix]).type(torch.float)
    y = torch.stack([data[i+1:i+args.block_size+1] for i in ix]).type(torch.float)
    return x, y


def get_args():
    parser = argparse.ArgumentParser(description="Neural Prediction Arguments")
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--block_size', type=int)
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--d_model', type=int)
    parser.add_argument('--n_heads', type=int)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--bias', type=bool)
    parser.add_argument('--tvt_tuple', type=tuple)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--device', type=torch.device)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--step_tuple', type=tuple)
    parser.add_argument('--criterion', type=str, choices=['L1', 'MSE'])
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--save_dir', type=str)
    parser.set_defaults(batch_size = 32,
                        block_size = 8,
                        vocab_size = 5,
                        dropout = 0.1,
                        d_model = 128,
                        n_heads = 4,
                        n_layers = 3,
                        bias = True,
                        tvt_tuple = (0.7, 0.15, 0.15),
                        lr = 3e-4,
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        data_dir = r"D:\Data\Research\NEURO\movement\processed_data\neural_dataset.pkl",
                        step_tuple = (1, 1, 1),
                        criterion = 'L1',
                        warmup_steps = 100,
                        n_epochs = 1,
                        save_dir = r"C:\Users\jackm\PycharmProjects\NEURO\NEURO_MOVEMENT\models\saved_models/"
                        )
    return parser.parse_args()


def training():
    # import args
    args = get_args()

    with open(args.data_dir, 'rb') as f:
        neural_data = pickle.load(f)
    f.close()
    num_samples, num_neurons = neural_data.shape
    args.vocab_size = num_neurons
    tvt_data = split_data(neural_data, args.tvt_tuple)
    args.step_tuple = (len(tvt_data[0]) // args.batch_size, len(tvt_data[1]) // args.batch_size,
                       len(tvt_data[2]) // args.batch_size)
    # model stuff
    model = Transformer(args.vocab_size, args.d_model, args.n_heads, args.dropout, args.n_layers, args.bias,
                        args.block_size, args.device)
    model = model.to(args.device)
    print(round(sum(p.numel() for p in model.parameters()) / 1e6, 2), 'M parameters')
    criterion = None
    if args.criterion == 'L1':
        criterion = nn.L1Loss(reduction='mean')
    elif args.criterion == 'MSE':
        criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamOpt(args.d_model, lr_mult=1.0, warmup=args.warmup_steps, optimizer=optimizer)
    best_valid_loss = 100.0
    for epoch in tqdm(range(args.n_epochs), desc="EPOCH"):
        model.train()
        for train_step in range(args.step_tuple[0]):
            scheduler.zero_grad()
            xb, yb = get_batch('train', tvt_data, args)
            xb, yb = xb.to(args.device), yb.to(args.device)

            logits = model(xb)
            logits = logits[:, -1, :]
            yb = yb[:, -1, :]

            loss = criterion(logits, yb)
            loss.backward()
            scheduler.step_and_update()
        model.eval()
        running_valid_loss = 0.0
        with torch.no_grad():
            for valid_step in range(args.step_tuple[1]):
                xb, yb = get_batch('valid', tvt_data, args)
                xb, yb = xb.to(args.device), yb.to(args.device)

                logits = model(xb)

                logits = logits[:, -1, :]
                yb = yb[:, -1, :]
                valid_loss = criterion(logits, yb)
                running_valid_loss += valid_loss
            avg_valid_loss = running_valid_loss / (valid_step+1)
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                torch.save(model.state_dict(), args.save_dir + 'best_valid_model.pt')

    best_model = Transformer(args.vocab_size, args.d_model, args.n_heads, args.dropout, args.n_layers, args.bias,
                             args.block_size, args.device)
    best_model = best_model.to(args.device)
    best_model.load_state_dict(torch.load(args.save_dir + 'best_valid_model.pt'))
    best_model.eval()
    running_test_loss = 0.0
    for test_step in range(args.step_tuple[2]):
        xb, yb = get_batch('test', tvt_data, args)
        xb, yb = xb.to(args.device), yb.to(args.device)

        logits = model(xb)

        logits = logits[:, -1, :]
        yb = yb[:, -1, :]
        test_loss = criterion(logits, yb)
        running_test_loss += test_loss
    avg_test_loss = running_test_loss / (test_step+1)
    print(f"AVG TEST LOSS: {avg_test_loss}")
    xb, yb = get_gen_batch('test', tvt_data, args)
    xb, yb = xb.to(args.device), yb.to(args.device)
    out = best_model.generate(xb, 5)
    print(out.shape)


if __name__ == '__main__':
    training()