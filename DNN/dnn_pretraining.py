import argparse
import random
import sys

import optuna
import torch
import numpy as np
import torch.optim as optim
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
parser.add_argument('--act', type=str)
parser.add_argument('--dropout', type=float)
parser.add_argument('--lr', type=float)
parser.add_argument('--bias', type=bool)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--n_epochs', type=int)
parser.add_argument('--gamma', type=int)
parser.add_argument('--tvt_tuple', type=tuple)
parser.add_argument('--momentum', type = float)
parser.add_argument('--base_dir', type=str)
parser.add_argument('--model_save_dir', type=str)
parser.set_defaults(num_layers=4, width = 512, act = 'relu', dropout=0.3,
                    lr=1e-4, bias=True, batch_size=1, n_epochs=50,
                    gamma = 3, tvt_tuple = (0.8, 0.1, 0.1), momentum = 0.15,
                    base_dir = r"D:\Data\Research\NEURO\touch/",
                    model_save_dir = r'C:\Users\jackm\PycharmProjects\NEURO\DNN\saved_models/')

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


class GenericResidualDNN(nn.Module):
    def __init__(self, linear_stack):
        super(GenericResidualDNN, self).__init__()
        self.linear_stack = linear_stack

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits


class ResidualBlock(nn.Module):
    def __init__(self, width, act, dropout, bias):
        super(ResidualBlock, self).__init__()
        self.width = width
        self.act_fn = None
        if act == 'relu':
            self.act_fn = nn.ReLU()
        elif act == 'leakyrelu':
            self.act_fn = nn.LeakyReLU()
        self.dropout = dropout
        self.bias = bias
        self.main = nn.Sequential(
            nn.Linear(width, width, bias=self.bias),
            # nn.BatchNorm1d(width),
            self.act_fn,
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = x + self.main(x)
        return out


def define_resdnn(num_layers, width, num_inputs, act, dropout, num_outputs, bias):
    #layers = [nn.Linear(num_inputs, width, bias=bias), nn.BatchNorm1d(width),
    #         nn.LeakyReLU(lparam)]
    act_fn = None
    if act == 'relu':
        act_fn = nn.ReLU()
    elif act == 'leakyrelu':
        act_fn = nn.LeakyReLU()
    layers = [nn.Linear(num_inputs, width, bias=bias)]
    layers.append(act_fn)
    for i in range(num_layers - 1):
        layers.append(ResidualBlock(width, act, dropout, bias))

    layers.append(nn.Linear(width, num_outputs, bias=bias))

    stack = nn.Sequential(*layers)

    model = GenericResidualDNN(stack)

    return model


class FocalLoss(nn.Module):
    def __init__(self, alpha = None, gamma = 2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets.long()] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss


def pretraining(args):
    with open(args.base_dir + 'neuron_to_limb1_1h.pkl', 'rb') as f:
        DATA = pickle.load(f)
    f.close()
    INPUTS, TARGETS = DATA
    train_data, valid_data, test_data = split_data(DATA, args.tvt_tuple)
    num_samples, num_neurons = INPUTS.shape
    num_samples, num_classes = TARGETS.shape
    class_weights_pre = get_weights(TARGETS)
    class_weights = torch.FloatTensor(class_weights_pre)
    class_weights = class_weights.to(device)
    train_dataset = CustomNeuralData(train_data)
    valid_dataset = CustomNeuralData(valid_data)
    test_dataset = CustomNeuralData(test_data)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=True)
    print("Dataloaders DONE")
    model = define_resdnn(args.num_layers, args.width, num_neurons, args.lparam,
                          args.dropout, num_classes, bias=True)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
    wandb.init(
        project="NEURO_neurontolimb_DNN",

        config={
            "num_layers": args.num_layers,
            "width": args.width,
            "dropout": args.dropout,
            "lparam": args.lparam,
            "gamma": args.gamma,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "epochs": args.n_epochs
        }
    )

    best_valid_loss = 10000.0
    for epoch in range(args.n_epochs):
        model.train()
        for i, (x_train, y_train) in tqdm(enumerate(train_dataloader)):
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()

            y_pred = model(x_train)

            loss = criterion(y_pred, y_train)
            if i % 5 == 0:
                wandb.log({"train_loss": loss.item()})

            loss.backward()

            optimizer.step()
        model.eval()
        running_valid_loss = 0.0
        for j, (x_valid, y_valid) in tqdm(enumerate(valid_dataloader)):
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)
            y_pred_valid = model(x_valid)

            valid_loss = criterion(y_pred_valid, y_valid)

            if j % 5 == 0:
                wandb.log({"valid_loss": valid_loss.item()})

            running_valid_loss += valid_loss.item()
        running_valid_loss /= j
        wandb.log({"running_valid_loss": running_valid_loss})
        if running_valid_loss < best_valid_loss:
            torch.save(model.state_dict(), args.model_save_dir + 'best_valid_model.pt')
            best_valid_loss = running_valid_loss

    model = define_resdnn(args.num_layers, args.width, num_neurons, args.lparam,
                          args.dropout, num_classes, bias=True)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_save_dir + "best_valid_model.pt"))
    running_test_loss = 0.0
    with torch.no_grad():
        model.eval()
        for k, (x_test, y_test) in tqdm(enumerate(test_dataloader)):
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_pred_test = model(x_test)

            test_loss = criterion(y_pred_test, y_test)

            if k % 5 == 0:
                wandb.log({"test_loss": test_loss.item()})
            running_test_loss += test_loss.item()
    wandb.log({"running_test_loss": running_test_loss})
    return running_test_loss


def tuning(trial):
    # set parameters
    args = parser.parse_args()
    args.num_layers = trial.suggest_int("num_layers", 1, 4)
    args.width = trial.suggest_categorical("width", [128, 256, 512, 1024])
    args.dropout = trial.suggest_float("dropout", 0.0, 0.9)
    args.gamma = trial.suggest_categorical("gamma", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    args.momentum = trial.suggest_float("momentum", 0.0, 0.5)
    args.lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    args.act = trial.suggest_categorical("act", ['relu', 'leakyrelu'])

    with open(args.base_dir + 'neuron_to_limb1_1h.pkl', 'rb') as f:
        DATA = pickle.load(f)
    f.close()
    INPUTS, TARGETS = DATA
    train_data, valid_data, test_data = split_data(DATA, args.tvt_tuple)
    num_samples, num_neurons = INPUTS.shape
    num_samples, num_classes = TARGETS.shape
    class_weights_pre = get_weights(TARGETS)
    class_weights = torch.FloatTensor(class_weights_pre)
    class_weights = class_weights.to(device)
    train_dataset = CustomNeuralData(train_data)
    valid_dataset = CustomNeuralData(valid_data)
    test_dataset = CustomNeuralData(test_data)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=True)
    model = define_resdnn(args.num_layers, args.width, num_neurons, args.act,
                          args.dropout, num_classes, bias=True)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
    wandb.init(
        project="NEURO_TOUCH_DNN_TUNING",

        config={
            "num_layers": args.num_layers,
            "width": args.width,
            "dropout": args.dropout,
            "act": args.act,
            "gamma": args.gamma,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "epochs": args.n_epochs,
            "lr": args.lr
        }
    )

    best_valid_loss = 10000.0
    for epoch in range(args.n_epochs):
        model.train()
        for i, (x_train, y_train) in tqdm(enumerate(train_dataloader)):
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()

            y_pred = model(x_train)

            loss = criterion(y_pred, y_train)
            if i % 5 == 0:
                wandb.log({"train_loss": loss.item()})

            loss.backward()

            optimizer.step()
        model.eval()
        running_valid_loss = 0.0
        for j, (x_valid, y_valid) in tqdm(enumerate(valid_dataloader)):
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)
            y_pred_valid = model(x_valid)

            valid_loss = criterion(y_pred_valid, y_valid)

            if j % 5 == 0:
                wandb.log({"valid_loss": valid_loss.item()})

            running_valid_loss += valid_loss.item()
        running_valid_loss /= j
        wandb.log({"running_valid_loss": running_valid_loss})
        if running_valid_loss < best_valid_loss:
            torch.save(model.state_dict(), args.model_save_dir + f'best_valid_model{trial.number}.pt')
            best_valid_loss = running_valid_loss

    model = define_resdnn(args.num_layers, args.width, num_neurons, args.act,
                          args.dropout, num_classes, bias=True)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_save_dir + f'best_valid_model{trial.number}.pt'))
    running_test_loss = 0.0
    with torch.no_grad():
        model.eval()
        for k, (x_test, y_test) in tqdm(enumerate(test_dataloader)):
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_pred_test = model(x_test)

            test_loss = criterion(y_pred_test, y_test)

            if k % 5 == 0:
                wandb.log({"test_loss": test_loss.item()})
            running_test_loss += test_loss.item()
    wandb.log({"running_test_loss": running_test_loss})
    wandb.finish()
    return running_test_loss


def interpret_multiclass_results(args):


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
    model = define_resdnn(args.num_layers, args.width, num_neurons, args.lparam,
                          args.dropout, 5, bias=True)
    model = model.to(device)
    model.load_state_dict(torch.load(f"../Transformer/saved_models/ntldnn.pt"))
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


def get_weights(target_data):
    num_samples, num_classes = target_data.shape
    num_samples_per_class = [0 for _ in range(num_classes)]
    for i in range(num_samples):
        for j in range(num_classes):
            if target_data[i][j].item() == 1.0:
                num_samples_per_class[j] += 1
    class_weights = []
    for count in num_samples_per_class:
        weight = 1 / (count / num_samples)
        class_weights.append(weight)
    return class_weights


def tune_w_optuna(num_trials):
    study = optuna.create_study(study_name='DNN_NEURO_TOUCH_TUNING',
                                storage='sqlite:///neurotune1.db',
                                load_if_exists=True,
                                direction="minimize")
    study.optimize(tuning, n_trials=num_trials)


if __name__ == '__main__':
    tune_w_optuna(50)