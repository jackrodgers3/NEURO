import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class DM(nn.Module):
    def __init__(self, input_size, output_size, width, lparam):
        super(DM, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, width),
            nn.LeakyReLU(lparam),
            nn.Linear(width, width),
            nn.LeakyReLU(lparam),
            nn.Linear(width, output_size)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, data):
        self.inputs = data[0]
        self.outputs = data[1]

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, item):
        return self.inputs[item], self.outputs[item]


def train(config):
    input_data = torch.randn(size=(20000, 25), dtype=torch.float, requires_grad=True)
    output_data = torch.randint(0, 1, size=(20000, 3), dtype=torch.float, requires_grad=True)
    train_data = [input_data[:15000], output_data[:15000]]
    valid_data = [input_data[15000:], output_data[15000:]]
    train_dataset = CustomDataset(train_data)
    valid_dataset = CustomDataset(valid_data)
    train_dataloader = DataLoader(train_dataset, 32)
    valid_dataloader = DataLoader(valid_dataset, 32)
    model = DM(25, 3, config.width, config.lparam)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_valid_loss = 100.0
    for epoch in range(4):
        model.train()
        for i, (x_train, y_train) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            y_train_pred = model(x_train)
            loss = criterion(y_train_pred,y_train)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for j, (x_train_valid, y_train_valid) in enumerate(valid_dataloader):
                y_pred_valid = model(x_train_valid)
                valid_loss = criterion(y_pred_valid, y_train_valid)
                if valid_loss.mean() < best_valid_loss:
                    best_valid_loss = valid_loss.mean()
    return best_valid_loss


def main():
    wandb.init(entity='<entity>', config=sweep_configuration, project='my-first-sweep')
    cfg = wandb.config
    val_metrics = train(cfg)
    wandb.log({"val_loss": val_metrics})
    wandb.finish()

sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "lparam": {"values": [0.005, 0.01, 0.015, 0.02]},
        "lr": {"values": [1e-5, 5e-5, 1e-4, 5e-4]},
        "width": {"values": [128, 256, 512, 1024]}
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
wandb.agent(sweep_id, function=main, count=5, project="my-first-sweep")