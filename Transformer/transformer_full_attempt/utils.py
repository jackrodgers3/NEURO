import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from model import *
from cortex_pretraining import *


def test():
    args = get_args()
    with open(r"D:\Data\Research\NEURO\touch\neuron_to_limb1.pkl", 'rb') as f:
        full_data = pickle.load(f)
    f.close()
    raw_data = full_data[0]
    xb, yb = get_batch(raw_data, args)
    print(xb.shape, yb.shape)
    plot_2d_neurons(xb)


def plot_2d_neurons(neural_tensor):
    neuron_array = np.array(neural_tensor)
    print(neuron_array.shape)
    t = [i for i in range(1, 1720)]
    plt.plot(t, neuron_array[0], 'r')
    plt.title('Neuron vs. Spk')
    plt.xlabel('Neuron')
    plt.ylabel('Relative spk')
    plt.savefig('neuron2d.png', format='png')


if __name__ == '__main__':
    with open(r"D:\Data\Research\NEURO\touch\dneuron_to_limb1.pkl", 'rb') as f:
        data = pickle.load(f)
    f.close()
    inputs, outputs = data
    print(inputs.shape, outputs.shape)
    new_outputs = []
    for i in range(len(outputs)):
        newrow = [0.0, 0.0, 0.0, 0.0, 0.0]
        newrow[int(outputs[i].item())] = 1.0
        new_outputs.append(newrow)
    oh_outputs = torch.tensor(data=new_outputs, dtype=torch.float)
    print(oh_outputs.shape)
    print(oh_outputs[0])
    new_data = [inputs, oh_outputs]
    with open(r"D:\Data\Research\NEURO\touch\dneuron_to_limb1_1h.pkl", 'wb') as g:
        pickle.dump(new_data, g)
    g.close()