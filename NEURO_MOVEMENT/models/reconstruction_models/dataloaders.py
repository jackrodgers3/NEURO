import pickle
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from torch.utils.data import Dataset
import os
import uproot
torch.manual_seed(23)
base_dir = r'D:\Data\Research\NEURO\movement\processed_data/'


def getfirst_last(arr, x):
    first = -1
    last = -1
    for i in range(len(arr)):
        if arr[i] != x:
            continue
        if first == -1:
            first = i
            last = i
        else:
            last = i
    return first, last


class NeuroTransformerDataset(Dataset):
    def __init__(self, raw_data, block_size, delta = True, standardize = True, for_transformer = True):
        self.neu, self.intermediate, self.limb = raw_data
        self.standardize = standardize
        self.delta = delta
        self.ft = for_transformer
        self.block_size = block_size

        # process data
        self.neu = self.neu.transpose()
        self.limb = self.limb.transpose()
        num_points, num_neurons = self.neu.shape
        num_points2, num_limbs = self.limb.shape
        self.inputs = np.empty(shape=(num_points, num_neurons))
        self.targets = np.empty(shape=(num_points, num_limbs))
        if delta:
            for c in range(1, num_points):
                self.inputs[c] = self.neu[c] - self.neu[c - 1]
                inter_val_f1, inter_val_l1 = getfirst_last(self.intermediate, c - 1)
                inter_val_f2, inter_val_l2 = getfirst_last(self.intermediate, c)
                self.targets[c] = self.limb[inter_val_l2] - self.limb[inter_val_f1]
        else:
            for c in range(0, num_points):
                self.inputs[c] = self.neu[c]
                inter_val_f, inter_val_l = getfirst_last(self.intermediate, c)
                self.targets[c] = self.limb[inter_val_f]
        self.inputs = torch.from_numpy(self.inputs)
        self.targets = torch.from_numpy(self.targets)

        if self.standardize:
            self.neu_means = torch.mean(self.inputs, axis = 0)
            self.neu_stdevs = torch.std(self.inputs, 0, True)
            self.inputs = (self.inputs - self.neu_means) / self.neu_stdevs
            self.limb_means = torch.mean(self.targets, axis=0)
            self.limb_stdevs = torch.std(self.targets, 0, True)
            self.targets = (self.targets - self.limb_means) / self.limb_stdevs

        if self.ft:
            self.tok_inputs = np.empty()

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def get_data(self):
        return self.inputs, self.targets

    def __len__(self):
        return len(self.inputs)

    def get_shapes(self):
        return self.inputs.shape, self.targets.shape


def plot_distributions(dataset):
    inps, tgts = dataset.get_data()
    inps = inps.detach().cpu().numpy().flatten()
    tgts = tgts.detach().cpu().numpy().flatten()
    plt.hist(inps)
    plt.title('Neural Distribution')
    plt.xlabel('Neural Spk')
    plt.ylabel('Density')
    plt.show()
    plt.cla()
    plt.clf()
    plt.hist(tgts)
    plt.title('Limb Distribution')
    plt.xlabel('Limb coords')
    plt.ylabel('Density')
    plt.show()

if __name__ == '__main__':
    pass