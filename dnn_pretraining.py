import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pickle
import math
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import wandb

parser = argparse.ArgumentParser('DNN PreTraining Arguments')
parser.add_argument('--num_layers', type=int)
parser.add_argument('--width', type=int)
parser.add_argument('--lparam', type=float)
parser.add_argument('--dropout', type=float)
parser.add_argument('--lr', type=float)
parser.add_argument('--bias', type=bool)
parser.add_argument('--base_dir', type = str)
parser.add_argument('--weighted', type=bool)
parser.add_argument('--tvt_split', type=tuple)
parser.add_argument('--label_smothing', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--n_epochs', type=int)
parser.set_defaults(num_layers=4, width = 1024, lparam=0.01, dropout=0.25,
                    lr=3e-4, bias=True, base_dir=r'C:\Users\jackm\PycharmProjects\NEURO/',
                    weighted = True, tvt_split = (0.7, 0.1, 0.2),
                    label_smoothing = 0.05, batch_size=64, n_epochs=20)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomNeuralData(Dataset):
    def __init__(self, data):
        self.inputs = data[0]
        self.targets = data[1]
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def split_data(full_dataset, tvt_tuple):
    cut1 = math.ceil(len(full_dataset[0]) * tvt_tuple[0])
    cut2 = math.ceil(len(full_dataset[0]) * (tvt_tuple[0] + tvt_tuple[1]))
    train_data = [full_dataset[0][:cut1], full_dataset[1][:cut1]]
    valid_data = [full_dataset[0][cut1:cut2], full_dataset[1][cut1:cut2]]
    test_data = [full_dataset[0][cut2:], full_dataset[1][cut2:]]
    return train_data, valid_data, test_data


class GenericDNN(nn.Module):
    def __init__(self, linear_stack, multiclass=True):
        super(GenericDNN, self).__init__()
        self.linear_stack = linear_stack
        self.multiclass = multiclass

    def forward(self, x):
        logits = self.linear_stack(x)
        if self.multiclass:
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits)
        return logits


class ResidualBlock(nn.Module):
    def __init__(self, width, lparam, dropout, bias):
        super(ResidualBlock, self).__init__()
        self.width = width
        self.lparam = lparam
        self.dropout = dropout
        self.bias = bias
        self.main = nn.Sequential(
            nn.Linear(width, width, bias=self.bias),
            nn.Dropout(dropout),
            nn.BatchNorm1d(width),
            nn.LeakyReLU(lparam)
        )

    def forward(self, x):
        out = x + self.main(x)
        return out


def define_resdnn(num_layers, width, num_inputs, lparam, dropout, num_outputs, multiclass, bias):
    layers = [nn.Linear(num_inputs, width, bias=bias), nn.BatchNorm1d(width),
              nn.LeakyReLU(lparam)]
    for i in range(num_layers - 1):
        layers.append(ResidualBlock(width, lparam, dropout, bias))

    layers.append(nn.Linear(width, num_outputs, bias=bias))

    stack = nn.Sequential(*layers)

    model = GenericDNN(stack, multiclass)

    return model


def pretraining(args):
    DATA = []
    with open(args.base_dir + 'inputs_pickle', 'rb') as f:
        INPUTS = pickle.load(f)
    f.close()

    with open(args.base_dir + 'targets_pickle', 'rb') as g:
        TARGETS = pickle.load(g)
    g.close()
    TARGETS = TARGETS[1:]
    num_neurons = 1719
    print(TARGETS.shape)
    TARGETSOH = torch.empty(size=(47978, 5))

    for i in range(len(TARGETS)):
        newrow = [0.0, 0.0, 0.0, 0.0, 0.0]
        newrow[int(TARGETS[i].item())] = 1.0

        TARGETSOH[i] = torch.tensor(data=newrow, dtype=torch.float)
    DATA.append(INPUTS)
    DATA.append(TARGETSOH)
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
    model = define_resdnn(args.num_layers, args.width, num_neurons, args.lparam,
                          args.dropout, 5, multiclass=True, bias=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    wandb.init(
        project="NEURO_neurontolimb_DNN",

        config={
            "label_smoothing": args.label_smoothing,
            "num_layers": args.num_layers,
            "width": args.width,
            "dropout": args.dropout,
            "lparam": args.lparam,
            "weighted": args.weighted
        }
    )

    best_valid_loss = 100.0
    for epoch in range(args.n_epochs):
        model.train()
        for i, (x_train, y_train) in tqdm(enumerate(train_dataloader)):
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()

            y_pred = model(x_train)

            loss = criterion(y_pred, y_train)

            if i % 2 == 0:
                wandb.log({"train_loss": loss.mean()})

            loss.backward()

            optimizer.step()

        model.eval()
        for j, (x_valid, y_valid) in tqdm(enumerate(valid_dataloader)):
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)

            y_pred_valid = model(x_valid)

            valid_loss = criterion(y_pred_valid, y_valid)

            if j % 2 == 0:
                wandb.log({"valid_loss": valid_loss.mean()})

            if valid_loss.mean() < best_valid_loss:
                best_valid_loss = valid_loss.mean()
                torch.save(model.state_dict(), f"Transformer/saved_models/ntldnn.pt")
    model = define_resdnn(args.num_layers, args.width, num_neurons, args.lparam,
                          args.dropout, 5, multiclass=True, bias=True)
    model = model.to(device)
    model.load_state_dict(torch.load(f"Transformer/saved_models/ntldnn.pt"))
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
    DATA = []
    with open(args.base_dir + 'delta_inputs_pickle', 'rb') as f:
        INPUTS = pickle.load(f)
    f.close()
    with open(args.base_dir + 'targets_pickle', 'rb') as g:
        TARGETS = pickle.load(g)
    g.close()
    TARGETS = TARGETS[1:]
    NEWTARGETS = torch.empty(size=(47977, 5), dtype=torch.float)
    indices = [[], [], [], [], []]
    for j in range(len(TARGETS)):
        newrow = [0.0, 0.0, 0.0, 0.0, 0.0]
        newrow[int(TARGETS[j].item())] = 1.0
        indices[int(TARGETS[j].item())].append(j)
        NEWTARGETS[j] = torch.tensor(newrow)
    print(indices)
    num_neurons = 1719
    DATA.append(INPUTS)
    DATA.append(NEWTARGETS)

    print(f"NUM_SAMPLES, INPUT SIZE: {DATA[0].shape[0], DATA[0].shape[1]}")
    print(f"NUM SAMPLES, OUTPUT SIZE: {DATA[1].shape[0], DATA[1].shape[1]}")
    train_dataset = CustomNeuralData(DATA)
    test_dataloader = DataLoader(train_dataset, train_dataset.__len__(), shuffle=True, pin_memory=True)
    if args.weighted:
        # getting class counts
        class_counts = [0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(len(TARGETS)):
            class_counts[int(TARGETS[i].item())] += 1.0
        class_weights = [1.0/class_counts[i] for i in range(len(class_counts))]
        sampler = WeightedRandomSampler(weights=class_weights, num_samples=train_dataset.__len__(), replacement=True)
        test_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=train_dataset.__len__())
    print("Dataloaders DONE")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    model = define_resdnn(args.num_layers, args.width, num_neurons, args.lparam,
                          args.dropout, 5, multiclass=True, bias=True)
    model = model.to(device)
    model.load_state_dict(torch.load(f"Transformer/saved_models/ntldnn.pt"))
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
    interpret_multiclass_results(args)