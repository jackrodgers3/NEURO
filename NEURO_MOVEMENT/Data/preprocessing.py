import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BASE_DIR = r"D:\Data\Research\NEURO\movement\Animal1_Movement/"
SAVE_DIR = r"D:\Data\Research\NEURO\movement\processed_data/"

def load_npy_files(directory):
    #Made by Anthony Aportela
    data_dict = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.npy'):
            file_path = os.path.join(directory, file_name)
            data = np.load(file_path, allow_pickle=True)
            key = os.path.splitext(file_name)[0]  # Extract the file name without extension as the dictionary key
            data_dict[key] = data
    return data_dict


def create_neuron_limb_coord_data(data_dict, save_dir, round_pix = False, normalized = False):
    base_filename = 'neuron_limb_coord_base_data'
    DATA = []

    neurons = torch.empty(size=(36660, 3327), dtype=torch.float)
    limb_coords = torch.empty(size=(36660, 8), dtype=torch.float)

    limb_coord_raw = data_dict['behav_coord_likeli']
    intermediate_map = data_dict['idx_coord_neural']
    neuron_spks_raw = data_dict['spks_final']

    # first limb coord tensor
    for i in tqdm(range(36660), desc="limb tensor"):
        new_tens = []
        for j in range(8):
            if round_pix:
                new_tens.append(round(limb_coords[j][i]))
            else:
                new_tens.append(limb_coord_raw[j][i])
        limb_coords[i] = torch.tensor(data=new_tens, dtype=torch.float)

    # now neuron spk tensor

    for j in tqdm(range(36660), desc="neuron tensor"):
        new_tens = []
        for k in range(3327):
            new_tens.append(neuron_spks_raw[k][intermediate_map[j]])
        neurons[j] = torch.tensor(data=new_tens, dtype=torch.float)

    DATA.append(neurons)
    DATA.append(limb_coords)
    for i in range(len(DATA)):
        print(f"Data {i} shape: {DATA[i].shape}")
    if round_pix:
        base_filename = base_filename + '_rounded'
    if normalized:
        base_filename = base_filename +  '_normed'
    with open(save_dir + base_filename + '.pkl', 'wb') as f:
        pickle.dump(DATA, f)
    f.close()


def create_delta_neuron_limb_coord_data(data_dict, save_dir, delta_input = True, delta_output = True, prob_based = True):
    base_filename = 'base_data'
    DATA = []

    neurons = torch.empty(size=(36660, 3327), dtype=torch.float)
    limb_coords = torch.empty(size=(36660, 8), dtype=torch.float)

    limb_coord_raw = data_dict['behav_coord_likeli']
    intermediate_map = data_dict['idx_coord_neural']
    neuron_spks_raw = data_dict['spks_final']

    # first limb coord tensor
    for i in tqdm(range(36660), desc="limb tensor"):
        new_tens = []
        for j in range(8):
            new_tens.append(limb_coord_raw[j][i])
        limb_coords[i] = torch.tensor(data=new_tens, dtype=torch.float)

    # now neuron spk tensor

    for j in tqdm(range(36660), desc="neuron tensor"):
        new_tens = []
        for k in range(3327):
            new_tens.append(neuron_spks_raw[k][intermediate_map[j]])
        neurons[j] = torch.tensor(data=new_tens, dtype=torch.float)

    dneurons = torch.empty(size=(36659, 3327), dtype= torch.float)
    dlimb_coords = torch.empty(size=(36659, 8), dtype=torch.float)

    if delta_input:
        for i in tqdm(range(len(neurons)-1), desc="dneuron tensor"):
            dneurons[i] = neurons[i+1] - neurons[i]
        DATA.append(dneurons)
        base_filename = base_filename + '_dneuron'
    if delta_output:
        for i in tqdm(range(len(limb_coords)-1), desc="dlimb tensor"):
            dlimb_coords[i] = limb_coords[i+1] - limb_coords[i]
        if prob_based:
            dlimb_coords = torch.abs(dlimb_coords)
            dlimb_coords_prob = torch.zeros_like(dlimb_coords, dtype=torch.float)
            for i in range(len(dlimb_coords)):
                dlimb_coords_prob[i][torch.argmax(dlimb_coords[i]).item()] = 1.0
                print(dlimb_coords_prob[i])
            DATA.append(dlimb_coords_prob)
            base_filename = base_filename + '_dlimb_coords_prob'
        else:
            DATA.append(dlimb_coords)
            base_filename = base_filename + '_dlimb_coords'

    for i in range(len(DATA)):
        print(f"Data {i} shape: {DATA[i].shape}")

    with open(save_dir + base_filename + '.pkl', 'wb') as f:
        pickle.dump(DATA, f)
    f.close()


def create_transformer_dataset():
    with open(r"D:\Data\Research\NEURO\movement\processed_data\neuron_limb_coord_base_data.pkl", 'rb') as f:
        DATA = pickle.load(f)
    f.close()
    neurons, limbs = DATA
    neurons = neurons.numpy()
    limbs = limbs.numpy()
    num_samples, num_neurons = neurons.shape
    num_samples, num_limb_coords = limbs.shape
    print("Number of neurons:", num_neurons, "Number of limb coords:", num_limb_coords)
    dneurons_embd = np.empty(shape=(num_samples-1, num_neurons))
    limbs_embd = np.empty(shape=(num_samples-1, num_limb_coords))
    for i in tqdm(range(num_samples-1)):
        dneurons_embd[i] = neurons[i+1] - neurons[i]
        limbs_embd[i] = limbs[i+1] - limbs[i]
    dneurons_embd = np.around(dneurons_embd, decimals=0)
    limbs_embd = np.around(limbs_embd, decimals=0)
    '''
    neurons_plot_arr = np.ravel(dneurons_embd)
    limbs_plot_arr = np.ravel(limbs_embd)
    fig = plt.figure()
    n_bins = 40
    plt.hist(neurons_plot_arr, bins = n_bins)
    plt.yscale('log')
    plt.title("Neuron Dist.")
    plt.show()
    plt.cla()
    plt.clf()
    plt.hist(limbs_plot_arr, bins = n_bins)
    plt.yscale('log')
    plt.title("Limb coords dist.")
    plt.show()
    plt.cla()
    plt.clf()
    '''

    ins = torch.from_numpy(dneurons_embd).type(torch.IntTensor)
    targets = torch.from_numpy(limbs_embd).type(torch.IntTensor)
    vocabs = (np.max(dneurons_embd) - np.min(dneurons_embd), np.max(limbs_embd) - np.min(limbs_embd))
    DATA = [ins, targets, vocabs]
    print(vocabs)
    with open(SAVE_DIR + 'transformer_data.pkl', 'wb') as f:
        pickle.dump(DATA, f)
    f.close()


if __name__ == '__main__':
    create_transformer_dataset()
