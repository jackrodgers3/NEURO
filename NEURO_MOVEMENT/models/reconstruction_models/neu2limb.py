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
    def __init__(self, d_model, head_size, bias, dropout, block_size, masked):
        super(Head, self).__init__()
        self.key = nn.Linear(d_model, head_size, bias)
        self.query = nn.Linear(d_model, head_size, bias)
        self.value = nn.Linear(d_model, head_size, bias)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.masked = masked
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_k, x_q, x_v):
        B, T, C = x_k.shape
        k = self.key(x_k)
        q = self.query(x_q)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        if self.masked:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x_v)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, head_size, bias, dropout, block_size, masked):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(d_model, head_size, bias, dropout, block_size, masked) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_k, x_q, x_v):
        out = torch.cat([h(x_k, x_q, x_v) for h in self.heads], dim=-1)
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


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout, bias, block_size):
        super(EncoderLayer, self).__init__()
        d_k = d_model // n_head
        self.ffwd = FeedForward(d_model, dropout)
        self.sa = MultiHeadAttention(d_model, n_head, d_k, bias, dropout, block_size, masked=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.sa(x, x, x))
        x = self.ln2(x + self.ffwd(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout, bias, block_size):
        super(DecoderLayer, self).__init__()
        d_k = d_model // n_head
        self.ffwd = FeedForward(d_model, dropout)
        self.sa = MultiHeadAttention(d_model, n_head, d_k, bias, dropout, block_size, masked=True)
        self.ca = MultiHeadAttention(d_model, n_head, d_k, bias, dropout, block_size, masked=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, enc_output, x):
        x = self.ln1(x + self.sa(x, x, x))
        x = self.ln2(x + self.ca(x, enc_output, x))
        x = self.ln3(x + self.ffwd(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size1, vocab_size2, d_model, n_head, dropout, n_layer, bias, block_size, device):
        super(Transformer, self).__init__()
        self.token_embedding1 = nn.Linear(vocab_size1, d_model, bias)
        self.pos_embedding1 = nn.Embedding(block_size, d_model)
        self.token_embedding2 = nn.Linear(vocab_size2, d_model, bias)
        self.pos_embedding2 = nn.Embedding(block_size, d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_head, dropout, bias, block_size) for _ in range(n_layer)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_head, dropout, bias, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size2)
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

    def forward(self, inputs, targets):
        B1, T1, C1 = inputs.shape
        B2, T2, C2 = targets.shape
        # print(B, T)
        tok_emb1 = self.token_embedding1(inputs)
        pos_emb1 = self.pos_embedding1(torch.arange(T1, device=self.device))
        tok_emb2 = self.token_embedding2(targets)
        pos_emb2 = self.pos_embedding2(torch.arange(T2, device=self.device))
        x1 = tok_emb1 + pos_emb1
        x2 = tok_emb2 + pos_emb2
        enc_output = x1
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)

        dec_output = x2
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(enc_output, dec_output)
        x = self.ln_f(dec_output)
        logits = self.lm_head(x)
        return logits


if __name__ == '__main__':
    pass