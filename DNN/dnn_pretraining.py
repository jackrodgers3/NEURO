import argparse
import random
import sys
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
parser.add_argument('--label_smothing', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--n_epochs', type=int)
parser.add_argument('--base_dir', type=str)
parser.set_defaults(num_layers=3, width = 1024, lparam=0.01, dropout=0.3,
                    lr=1e-4, bias=True,
                    label_smoothing = 0.05, batch_size=32, n_epochs=50,
                    base_dir = r'C:\Users\jackm\PycharmProjects\NEURO\Transformer/')

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
    def __init__(self, linear_stack, multiclass=True):
        super(GenericResidualDNN, self).__init__()
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

    model = GenericResidualDNN(stack, multiclass)

    return model


def pretraining(args):
    DATA = []
    with open(args.base_dir + 'inputs_pickle', 'rb') as f:
        INPUTS = pickle.load(f)
    f.close()
    with open(args.base_dir + 'targets_pickle', 'rb') as g:
        TARGETS = pickle.load(g)
    g.close()
    DATA.append(INPUTS)
    DATA.append(TARGETS)
    make_balanced_data(DATA, 5, 500)
    with open(r'C:\Users\jackm\PycharmProjects\NEURO\DNN\dnn_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    f.close()
    TARGETS1H = torch.empty(size=(len(TARGETS), 5), dtype=torch.float)
    for i in range(len(TARGETS)):
        newrow = [0.0 for j in range(5)]
        newrow[int(TARGETS[i].item())] = 1.0
        TARGETS1H[i] = torch.tensor(newrow, dtype=torch.float)
    print(f'1h shape: {TARGETS1H.shape}')
    test_data = [INPUTS, TARGETS1H]
    train_dataset = CustomNeuralData(train_data)
    test_dataset = CustomNeuralData(test_data)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=True, pin_memory=True)
    ex_input, ex_output = train_dataset.__getitem__(0)
    num_neurons, num_targets = ex_input.shape[0], ex_output.shape[0]
    print(f'Input shape: {num_neurons}\nOutput shape: {num_targets}')
    print("Dataloaders DONE")
    model = define_resdnn(args.num_layers, args.width, num_neurons, args.lparam,
                          args.dropout, num_targets, multiclass=True, bias=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    wandb.init(
        project="NEURO_neurontolimb_DNN",

        config={
            "label_smoothing": args.label_smoothing,
            "num_layers": args.num_layers,
            "width": args.width,
            "dropout": args.dropout,
            "lparam": args.lparam
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
        with torch.no_grad():
            for j, (x_test, y_test) in tqdm(enumerate(test_dataloader)):
                x_test, y_test = x_test.to(device), y_test.to(device)
                y_pred_test = model(x_test)
                test_loss = criterion(y_pred_test, y_test)
                if j % 2 == 0:
                    wandb.log({"valid_loss": test_loss.mean()})
                if test_loss.mean() < best_valid_loss:
                    best_valid_loss = test_loss.mean()
    # plot confusion matrix
    model.eval()
    val_predictions = model(test_data[0])
    top_pred_ids = val_predictions.argmax(axis=1).tolist()
    ground_truth_ids = test_data[1].argmax(axis=1).tolist()
    wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(
        preds=top_pred_ids, y_true=ground_truth_ids,
        class_names = [0, 1, 2, 3, 4]
    )})
    wandb.finish()
    return best_valid_loss


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


def make_balanced_data(raw_data, num_classes, min_counts):
    print(raw_data[0].shape)
    print(raw_data[1].shape)
    # get class counts and sort
    class_counts = [0 for b in range(num_classes)]
    class_indices = [[] for b in range(num_classes)]
    for i in range(len(raw_data[1])):
        class_counts[int(raw_data[1][i].item())] += 1
        class_indices[int(raw_data[1][i].item())].append(i)
    print(class_counts)
    min_count = min_counts
    # create master indices list
    master_indices = []
    for i in range(len(class_indices)):
        for j in range(min_count):
            master_indices.append(class_indices[i][j])
    # shuffle indices
    random.shuffle(master_indices)
    # create input and output data
    input_data = []
    output_data = []
    for i in range(len(master_indices)):
        ex_i = raw_data[0][master_indices[i]].tolist()
        ex_o = raw_data[1][master_indices[i]].tolist()
        input_data.append(ex_i)
        output_data.append(ex_o)
    # one hot encode outputs
    output_1h_data = []
    for i in range(len(output_data)):
        newrow = [0 for j in range(num_classes)]
        newrow[int(output_data[i])] = 1
        output_1h_data.append(newrow)
    inputs = torch.tensor(input_data, dtype=torch.float)
    outputs = torch.tensor(output_1h_data, dtype=torch.float)
    print(f'Input shape: {inputs.shape}\nOutput shape: {outputs.shape}')
    DATA = []
    DATA.append(inputs)
    DATA.append(outputs)
    with open(r'C:\Users\jackm\PycharmProjects\NEURO\DNN\dnn_data.pkl', 'wb') as f:
        pickle.dump(DATA, f)
    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    pretraining(args)