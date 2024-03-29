import math
import pickle
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from NEURO_MOVEMENT.models.classification_models import define_dnn
import argparse
import wandb
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


def get_args():
    parser = argparse.ArgumentParser("classification parser")

    parser.add_argument("--data_file", type = str)
    parser.add_argument("--tvt_tuple", type=tuple)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--act_fn", type=str)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--bias", type=bool)
    parser.add_argument("--residual", type=bool)
    parser.add_argument("--input_dim", type=int)
    parser.add_argument("--output_dim", type=int)
    parser.add_argument("--multiclass", type=bool)
    parser.add_argument("--lr", type = float)
    parser.add_argument("--criterion", type=str)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--momentum", type=float)

    parser.set_defaults(
        data_file = r"D:\Data\Research\NEURO\movement\processed_data\base_data_dneuron_dlimb_coords_prob.pkl",
        tvt_tuple = (0.7, 0.15, 0.15),
        batch_size = 16,
        n_layers = 4,
        hidden_dim = 2048,
        act_fn = 'leakyrelu',
        dropout = 0.2,
        bias = True,
        residual = False,
        input_dim = 1,
        output_dim = 1,
        multiclass = 'multiclass',
        lr = 1e-4,
        criterion = 'BCE',
        n_epochs = 50,
        save_dir = r"C:\Users\jackm\PycharmProjects\NEURO\NEURO_MOVEMENT\models\saved_models/",
        momentum = 0.1
    )

    return parser.parse_args()


def train():
    args = get_args()
    with open(args.data_file, 'rb') as f:
        full_data = pickle.load(f)
    f.close()
    num_samples, args.input_dim = full_data[0].shape
    num_samples, args.output_dim = full_data[1].shape
    train_data, valid_data, test_data = split_data(full_data, args.tvt_tuple)
    train_dataset, valid_dataset, test_dataset = CustomDataset(train_data), \
        CustomDataset(valid_data), CustomDataset(test_data)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    model = define_dnn(args.n_layers, args.input_dim, args.hidden_dim,
                       args.output_dim, args.act_fn, args.dropout,
                       args.bias, args.residual, args.multiclass)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)
    criterion = None
    if args.criterion == 'L1':
        criterion = nn.L1Loss(reduction='mean')
    elif args.criterion == 'MSE':
        criterion = nn.MSELoss(reduction='mean')
    elif args.criterion == 'BCE':
        criterion = nn.BCELoss(reduction='mean')

    wandb.init(
        project="DNN_NEURO",
        config={
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_layers": args.n_layers,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "residual": args.residual,
            "criterion": args.criterion
        }
    )

    best_valid_loss = float('inf')
    for epoch in tqdm(range(args.n_epochs), desc = "epoch"):
        model.train()
        for train_batch_id, (x_train, y_train) in enumerate(train_dataloader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            if train_batch_id % 5 == 0:
                wandb.log({"train_loss": loss.item()})
            loss.backward()
            optimizer.step()
        model.eval()
        running_valid_loss = 0.0
        for valid_batch_id, (x_valid, y_valid) in enumerate(valid_dataloader):
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)
            y_pred_valid = model(x_valid)
            valid_loss = criterion(y_pred_valid, y_valid)
            running_valid_loss += valid_loss.mean().item()
            if valid_batch_id % 5 == 0:
                wandb.log({"valid_loss": valid_loss.item()})
        wandb.log({"running_valid_loss": running_valid_loss})
        if running_valid_loss < best_valid_loss:
            best_valid_loss = running_valid_loss
            torch.save(model.state_dict(), args.save_dir + 'delta_class_model.pt')
    model = define_dnn(args.n_layers, args.input_dim, args.hidden_dim,
                       args.output_dim, args.act_fn, args.dropout,
                       args.bias, args.residual, args.multiclass)
    model = model.to(device)
    model.load_state_dict(torch.load(args.save_dir + 'delta_class_model.pt'))
    with torch.no_grad():
        model.eval()
        for test_batch_id, (x_test, y_test) in enumerate(test_dataloader):
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_pred_test = model(x_test)
            test_loss = criterion(y_pred_test, y_test)
            if test_loss % 5 == 0:
                wandb.log({"test_loss": test_loss.mean().item()})
    wandb.finish()


if __name__ == '__main__':
    train()


