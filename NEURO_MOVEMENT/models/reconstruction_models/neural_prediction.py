import pickle
import os
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import torch.nn.functional as F

with open(r"D:\Data\Research\NEURO\movement\processed_data\neural_dataset.pkl", 'rb') as f:
    neural_data = pickle.load(f)
f.close()
print(f"Data shape: {neural_data.shape}")
num_samples, num_neurons = neural_data.shape
# hyperparameters
batch_size = 16
block_size = 32
vocab_size = num_neurons
dropout = 0.2
d_model = 512
n_heads = 2
n_layers = 6
bias = True
tvt_tuple = (0.7, 0.15, 0.15)
lr = 3e-4
momentum = 0.1
max_iters = 200
eval_interval = 50
#################
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
n1 = int(tvt_tuple[0]*len(neural_data))
n2 = int((tvt_tuple[0] + tvt_tuple[1]) * len(neural_data))
train_data = neural_data[:n1]
valid_data = neural_data[n1:n2]
test_data = neural_data[n2:]


def get_batch(split):
    data = None
    if split == 'train':
        data = train_data
    elif split == 'valid':
        data = valid_data
    elif split == 'test':
        data = test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).type(torch.float)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).type(torch.float)
    return x, y


class Head(nn.Module):
    def __init__(self, d_model, head_size, bias, dropout):
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
    def __init__(self, d_model, num_heads, head_size, bias, dropout):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(d_model, head_size, bias, dropout) for _ in range(num_heads)])
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
    def __init__(self, d_model, n_head, dropout, bias):
        super(Block, self).__init__()
        d_k = d_model // n_head
        self.ffwd = FeedForward(d_model, dropout)
        self.sa = MultiHeadAttention(d_model, n_head, d_k, bias, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, dropout, n_layer, bias):
        super(Transformer, self).__init__()
        self.token_embedding = nn.Linear(vocab_size, d_model, bias)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_head, dropout, bias) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
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
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        #idx is (B, T)
        for _ in range(max_new_tokens):
            # crop idx to last tokens
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits = self(idx_cond)
            # focus on last time token
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax for probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# training
model = Transformer(vocab_size, d_model, n_heads, dropout, n_layers, bias)
m = model.to(device)
print(round(sum(p.numel() for p in m.parameters())/1e6, 2), 'M parameters')
criterion = nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

for iter in range(max_iters):
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    logits = model(xb)
    logits = logits[:, -1, :]
    yb = yb[:, -1, :]
    loss = criterion(logits, yb)
    if iter % 5 == 4:
        print(f"LOSS: {loss}")
    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()
    if iter % eval_interval == 0:
        running_valid_loss = 0.0
        m.eval()
        with torch.no_grad():
            for v in range(10):
                xb, yb = get_batch('valid')
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                logits = logits[:, -1, :]
                yb = yb[:, -1, :]
                valid_loss = criterion(logits, yb)
                running_valid_loss += valid_loss
        m.train()
        running_valid_loss /= 10
        print(f"Avg valid loss: {running_valid_loss}")
