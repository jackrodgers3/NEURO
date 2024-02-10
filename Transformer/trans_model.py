from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import copy
import matplotlib.pyplot as plt
import math as m
import numpy as np
from tqdm import tqdm
import sys
import argparse
from torch.optim import lr_scheduler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'CUDA? {torch.cuda.is_available()}')

###################SETUP########################
parser = argparse.ArgumentParser(description='TransModel PreTraining Arguments')
parser.add_argument("--lr", help="learning rate", type = float)
parser.add_argument("--batch_size", help="batch size", type = int)
parser.add_argument("--d_model", help="embedding dimension for model", type=int)
parser.add_argument("--h", help="number of SA heads in a MHA block", type=int)
parser.add_argument("--dropout", help="dropout rate", type = float)
parser.add_argument("--N", help="number of decoder layers", type = int)
parser.add_argument("--max_iters", help="Number of steps to train", type = int)
parser.add_argument("--eval_iters", help="Every eval_iters steps, loss will be validated", type=int)
parser.add_argument("--base_dir", help = "Directory to get stuff", type = str)
parser.add_argument("--save_dir", help="Directory to save output", type=str)
parser.add_argument("--rat", help="Train on rat 1 or 2", type = int)
parser.add_argument("--reproducibility", help="control randomness", type=bool)
parser.add_argument("--lossf", help="L1 is true, MSE is false", type = bool)

parser.set_defaults(lr=5e-3, batch_size=256, d_model=512, h=4, dropout=0.1,
                    N=8, max_iters=200, eval_iters=10,
                    base_dir=r'C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023\Transformer/',
                    save_dir=r'C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023\Output/',
                    rat=1, reproducibility = False, lossf = True)
args = parser.parse_args()

d_ff = 4 * args.d_model

if args.reproducibility:
    torch.manual_seed(25)

if args.rat == 1:
    with open(args.base_dir + 'inputs_pickle', 'rb') as f:
        INPUTS = pickle.load(f)
    f.close()

    with open(args.base_dir + 'targets_pickle', 'rb') as g:
        TARGETS = pickle.load(g)
    g.close()
    num_neurons = 1719
elif args.rat == 2:
    with open(args.base_dir + 'inputs2_pickle', 'rb') as f:
        INPUTS = pickle.load(f)
    f.close()

    with open(args.base_dir + 'targets2_pickle', 'rb') as g:
        TARGETS = pickle.load(g)
    g.close()
    num_neurons = 901
else:
    sys.exit()

##############FUNCTION/CLASS DEFS##########################


def get_lr(opt):
    for param in opt.param_groups:
        return param['lr']

def plot_lr(opt):
    sched1 = lr_scheduler.StepLR(optimizer=opt, step_size=args.max_iters//10,
                                gamma=0.6)
    sched2 = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=opt,
        T_0=args.max_iters//2,
        eta_min=1e-6)
    plt.title("Learning Rate Scheduler")
    plt.xlabel('Step')
    plt.ylabel('LR')
    plt.yscale('log')
    lrs = []
    for l in range(int(args.max_iters)):
        lrs.append(get_lr(opt))
        opt.step()
        if l <args.max_iters//2:
            sched1.step()
        else:
            sched2.step()
    plt.plot(np.arange(args.max_iters), np.array(lrs))
    plt.ylim(1e-10, np.max(lrs) * 10)
    plt.savefig(args.save_dir + 'lr_plot.png')
    plt.cla()
    plt.clf()

#base transformer model for neural encoding

def init_normal(sigma):
    def init_(tensor):
        return nn.init.normal_(tensor, mean = 0.0, std=sigma)
    return init_
'''
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
'''
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, bias = True):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias = bias)
        self.w_2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

def attention(query, key, value, dropout=None, attention_mult = 1.0):
    d_k = query.size(-1)

    scores = attention_mult * args.h * torch.matmul(query, key.transpose(-2, -1)) / d_k
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout, bias = True, attention_mult = 1.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = int(d_model // h)
        self.h = int(h)
        self.linears = clones(nn.Linear(d_model, d_model, bias = bias), 4)
        self.attention_mult = attention_mult
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):

        query, key, value = \
        [l(x).view(-1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self_attn = attention(query, key, value, dropout=self.dropout, attention_mult=self.attention_mult)

        x = x.transpose(1, 2).contiguous().view( -1, self.h * self.d_k)
        x = self.linears[-1](x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, size, feed_forward, self_attn, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.feed_forward = feed_forward
        self.self_attn = self_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer[1](x, self.feed_forward)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, y):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer[1](x, lambda x: self.self_attn(y, y, x))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N, ln_gain_mult):
        super(Decoder, self).__init__()
        self.ln_gain_mult = ln_gain_mult
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, elementwise_affine=True)

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x, y)
        return self.ln_gain_mult * self.norm(x)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, elementwise_affine= True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

#############SETUP########################
n = int(0.9*len(INPUTS))
train_inputs = INPUTS[:n]
train_targets = TARGETS[:n]
valid_inputs = INPUTS[n:]
valid_targets = TARGETS[n:]

def get_batch(split):
    #generate batch
    inputs = train_inputs if split == 'train' else valid_inputs
    targets = train_targets if split == 'train' else valid_targets
    if split == 'train':
        ix = torch.randint(len(train_inputs), (args.batch_size,))
    else:
        ix = torch.randint(len(valid_inputs), (args.batch_size,))
    x = torch.stack([inputs[i] for i in ix])
    y = torch.stack([targets[i] for i in ix])
    y = y.view(args.batch_size, 1)
    x.requires_grad = True
    y.requires_grad = True
    x, y = x.to(device), y.to(device)
    return x, y


######test###################


class NeuroTransformer(nn.Module):
    def __init__(self, input_size = 1, target_size = num_neurons, N = args.N, d_model = args.d_model, d_ff = d_ff,
                 h = args.h, dropout = args.dropout, bias = True, attention_mult = 1.0):
        super(NeuroTransformer, self).__init__()
        c = copy.deepcopy
        self.d_model = d_model
        self.bias = bias
        attn = MultiHeadAttention(h, d_model, attention_mult)
        ff = FeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, c(ff), c(attn), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), N, ln_gain_mult=1.0)

        self.input_positional_embedding = nn.Linear(input_size, d_model)

        self.target_positional_embedding = nn.Linear(target_size, d_model)

        self.reco_layer = nn.Linear(d_model, target_size)

    def forward(self, x, y):
        encoder_output = self.encoder(self.input_positional_embedding(x))
        decoder_input = self.target_positional_embedding(y)
        decoder_output = self.decoder(encoder_output, decoder_input)
        output = self.reco_layer(decoder_output)
        return output



model = NeuroTransformer().to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {pytorch_total_params}')


optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.98), eps=1e-8)
scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=args.max_iters//10, gamma=0.6)
scheduler2 = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=args.max_iters//2, eta_min=1e-6
    )
###FOR PLOTTING######
optimizer_test = torch.optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.98), eps=1e-8)
plot_lr(optimizer_test)

if args.lossf:
    criterion = nn.L1Loss()
else:
    criterion = nn.MSELoss()


#####SIMPLE TRAINING LOOP##########
steps = []
train_losses = []
valid_losses = []
best_validation_loss = 100.0
#xb: [B, num_neurons]
#yb: [B, 1]

start = timer()
model.train()
for iter in tqdm(range(1 ,args.max_iters+1)):
    '''
    #evaluate on train and validation sets
    if iter % eval_interval == 0:
        model.eval()
        X, Y = get_batch('valid')
        _, loss = model(Y)
        print(f"Valid loss: {loss}")
    '''

    xb, yb = get_batch('train')
    output = model(yb, xb)
    loss = criterion(output, xb)
    steps.append(iter)
    train_losses.append(loss.item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if iter < args.max_iters//2:
        scheduler.step()
    else:
        scheduler2.step()

    if iter % args.eval_iters == 0:
        model.eval()
        valids = []
        for i in tqdm(range(4798 // args.batch_size)):
            xb, yb = get_batch('valid')
            output = model(yb, xb)
            valid_loss = criterion(output, xb)
            valids.append(valid_loss.item())
        valid_losses.append(np.mean(valids))
        if np.mean(valids) < best_validation_loss:
            best_validation_loss = np.mean(valids)
            torch.save(model, args.save_dir + 'trans_model.pt')
        model.train()
end = timer()

total_time = end - start
print(f"Total training time: {total_time / 60.0} minutes")
print(f"Best Validation Loss: {best_validation_loss:.3f}")
plt.plot(steps, train_losses)
plt.xlabel('Steps')
plt.ylabel('L1 Loss')
plt.savefig(args.save_dir + 'train_trans.png')
plt.cla()
plt.clf()
plt.plot([i for i in range(0, args.max_iters//10)], valid_losses)
plt.xlabel(f'{args.eval_iters} step increment')
plt.ylabel('L1 Loss')
plt.savefig(args.save_dir + 'valid_trans.png')
#############################