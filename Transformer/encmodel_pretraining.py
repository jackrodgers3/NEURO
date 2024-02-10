from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import wandb
from tqdm import tqdm
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'CUDA? {torch.cuda.is_available()}')


class CustomNeuralData(Dataset):
    def __init__(self, data):
        self.inputs = data[0]
        self.targets = data[1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


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

    def update_learning_rate(self):
        self.steps += 1
        lr = self.lr_mult * self.get_lr_scale()
        for p in self.optimizer.param_groups:
            p['lr'] = lr


###################SETUP########################
parser = argparse.ArgumentParser(description='EncModel PreTraining Arguments')
parser.add_argument("--lr", help="learning rate", type = float)
parser.add_argument("--n_epochs", help="# of epochs to train", type=int)
parser.add_argument("--batch_size", help="batch size", type = int)
parser.add_argument("--d_model", help="embedding dimension for model", type=int)
parser.add_argument("--h", help="number of SA heads in a MHA block", type=int)
parser.add_argument("--dropout", help="dropout rate", type = float)
parser.add_argument("--N", help="number of decoder layers", type = int)
parser.add_argument("--base_dir", help = "Directory to get stuff", type = str)
parser.add_argument("--save_dir", help="Directory to save output", type=str)
parser.add_argument("--rat", help="Train on rat 1 or 2 or delta (3)", type = int)
parser.add_argument("--reproducibility", help="control randomness", type=bool)
parser.add_argument("--lossf", help="L1 is true, MSE is false", type = bool)
parser.add_argument("--tvt_split", help="train/valid/test split", type = tuple)
parser.add_argument("--weighted", help="Whether or not to weight classes", type=tuple)

parser.set_defaults(lr=5e-3, n_epochs = 5, batch_size=64, d_model=512, h=8, dropout=0.1,
                    N=8,
                    base_dir=r'C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023\Transformer/',
                    save_dir=r'C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023\Output/',
                    rat=3, reproducibility = True, lossf = True, tvt_split = (0.7, 0.1, 0.2), weighted = True)


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


def init_normal(sigma):
    def init_(tensor):
        return nn.init.normal_(tensor, mean = 0.0, std=sigma)
    return init_


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


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer[1](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N, ln_gain_mult):
        super(Decoder, self).__init__()
        self.ln_gain_mult = ln_gain_mult
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, elementwise_affine=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.ln_gain_mult * self.norm(x)


def split_data(full_dataset, tvt_tuple):
    cut1 = math.ceil(len(full_dataset[0]) * tvt_tuple[0])
    cut2 = math.ceil(len(full_dataset[0]) * (tvt_tuple[0] + tvt_tuple[1]))
    train_data = [full_dataset[0][:cut1], full_dataset[1][:cut1]]
    valid_data = [full_dataset[0][cut1:cut2], full_dataset[1][cut1:cut2]]
    test_data = [full_dataset[0][cut2:], full_dataset[1][cut2:]]
    return train_data, valid_data, test_data


################MODEL###################
class EncModel(nn.Module):
    def __init__(self, input_size, target_size, N, d_model, d_ff,
                 h, dropout, bias = True, attention_mult = 1.0):
        super(EncModel, self).__init__()
        c = copy.deepcopy
        self.d_model = d_model
        self.bias = bias
        attn = MultiHeadAttention(h, d_model, attention_mult)
        ff = FeedForward(d_model, d_ff, dropout)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), N, ln_gain_mult=1.0)

        self.input_positional_embedding = nn.Linear(input_size, d_model)

        self.reco_layer = nn.Linear(d_model, target_size)

    def forward(self, x):
        decoder_input = self.input_positional_embedding(x)
        decoder_output = self.decoder(decoder_input)
        output = self.reco_layer(decoder_output)
        return output


def pretraining(args):
    d_ff = 4 * args.d_model

    if args.reproducibility:
        torch.manual_seed(23)

    DATA = []
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
    elif args.rat == 3:
        with open(args.base_dir + 'delta_inputs_pickle', 'rb') as f:
            INPUTS = pickle.load(f)
        f.close()
        with open(args.base_dir + 'targets_pickle', 'rb') as g:
            TARGETS = pickle.load(g)
        g.close()
        TARGETS = TARGETS[1:]
        NEWTARGETS = torch.empty(size=(47977, 5), dtype=torch.float)
        for j in range(len(TARGETS)):
            newrow = [0.0, 0.0, 0.0, 0.0, 0.0]
            newrow[int(TARGETS[j].item())] = 1.0
            NEWTARGETS[j] = torch.tensor(newrow)
        num_neurons = 1719
    else:
        sys.exit()
    DATA.append(INPUTS)
    DATA.append(NEWTARGETS)

    print(f"NUM_SAMPLES, INPUT SIZE: {DATA[0].shape[0], DATA[0].shape[1]}")
    print(f"NUM SAMPLES, OUTPUT SIZE: {DATA[1].shape[0], DATA[1].shape[1]}")

    train_data, valid_data, test_data = split_data(DATA, args.tvt_split)
    train_dataset, valid_dataset, test_dataset = CustomNeuralData(train_data), \
        CustomNeuralData(valid_data), CustomNeuralData(test_data)
    if args.weighted:
        # getting class counts
        class_counts = [0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(len(TARGETS)):
            class_counts[int(TARGETS[i].item())] += 1.0
        class_weights = [1.0/class_counts[i] for i in range(len(class_counts))]
        sampler = WeightedRandomSampler(weights=class_weights, num_samples=train_dataset.__len__(), replacement=True)
        train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size)
    else:
        train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=True, pin_memory=True)
    print("Dataloaders DONE")

    model = EncModel(num_neurons, 5, args.N, args.d_model, d_ff, args.h,
                     args.dropout, True, 1.0)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = SeminalOpt(args.d_model, lr_mult=1.0, optimizer=optimizer, warmup=400)
    criterion = nn.CrossEntropyLoss()

    wandb.init(
        project="NEURO_neurontolimb_classification",

        config={
            "batch_size": args.batch_size,
            "num_layers": args.N,
            "d_model": args.d_model,
            "dropout": args.dropout,
            "h": args.h,
            "weighted": args.weighted
        }
    )

    best_valid_loss = 100.0
    for epoch in range(args.n_epochs):
        model.train()
        for i, (x_train, y_train) in tqdm(enumerate(train_dataloader)):
            x_train, y_train = x_train.to(device), y_train.to(device)

            scheduler.zero_grad()

            y_pred = model(x_train)

            loss = criterion(y_pred, y_train)

            if i % 2 == 0:
                wandb.log({"train_loss": loss.mean()})

            loss.backward()

            scheduler.step_and_update()

        model.eval()
        for j, (x_valid, y_valid) in tqdm(enumerate(valid_dataloader)):
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)

            y_pred_valid = model(x_valid)

            valid_loss = criterion(y_pred_valid, y_valid)

            if j % 2 == 0:
                wandb.log({"valid_loss": valid_loss.mean()})

            if valid_loss.mean() < best_valid_loss:
                best_valid_loss = valid_loss.mean()
                torch.save(model.state_dict(), f"saved_models/neutolimbclass.pt")
    model = EncModel(num_neurons, 5, args.N, args.d_model, d_ff, args.h,
                     args.dropout, True, 1.0)
    model = model.to(device)
    model.load_state_dict(torch.load(f"saved_models/neutolimbclass.pt"))
    model.eval()
    with torch.no_grad():
        for k, (x_test, y_test) in tqdm(enumerate(test_dataloader)):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_pred_test = model(x_test)
            test_loss = criterion(y_pred_test, y_test)
            wandb.log({"test_loss": test_loss.mean()})
    wandb.finish()


def interpret_multiclass_results(args):
    d_ff = args.d_model * 4
    DATA = []
    with open(args.base_dir + 'delta_inputs_pickle', 'rb') as f:
        INPUTS = pickle.load(f)
    f.close()
    with open(args.base_dir + 'targets_pickle', 'rb') as g:
        TARGETS = pickle.load(g)
    g.close()
    TARGETS = TARGETS[1:]
    NEWTARGETS = torch.empty(size=(47977, 5), dtype=torch.float)
    for j in range(len(TARGETS)):
        newrow = [0.0, 0.0, 0.0, 0.0, 0.0]
        newrow[int(TARGETS[j].item())] = 1.0
        NEWTARGETS[j] = torch.tensor(newrow)
    num_neurons = 1719
    DATA.append(INPUTS)
    DATA.append(NEWTARGETS)

    print(f"NUM_SAMPLES, INPUT SIZE: {DATA[0].shape[0], DATA[0].shape[1]}")
    print(f"NUM SAMPLES, OUTPUT SIZE: {DATA[1].shape[0], DATA[1].shape[1]}")

    train_data, valid_data, test_data = split_data(DATA, args.tvt_split)
    train_dataset, valid_dataset, test_dataset = CustomNeuralData(train_data), \
        CustomNeuralData(valid_data), CustomNeuralData(test_data)
    test_dataloader = DataLoader(test_dataset, test_dataset.__len__(), shuffle=True, pin_memory=True)
    print("Dataloaders DONE")
    criterion = nn.CrossEntropyLoss()
    model = EncModel(num_neurons, 5, args.N, args.d_model, d_ff, args.h,
                     args.dropout, True, 1.0)
    model = model.to(device)
    model.load_state_dict(torch.load(f"saved_models/neutolimbclass.pt"))
    model.eval()
    with torch.no_grad():
        for k, (x_test, y_test) in tqdm(enumerate(test_dataloader)):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_pred_test = model(x_test)
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred_test.argmax(axis=1))
    cm_df = pd.DataFrame(cm, index=['0', '1', '2', '3', '4'],
                         columns=['0', '1', '2', '3', '4'])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    pretraining(args)
    interpret_multiclass_results(args)


