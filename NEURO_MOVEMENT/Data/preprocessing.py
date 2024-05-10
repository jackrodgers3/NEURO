import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BASE_DIR = r"D:\Data\Research\NEURO\movement\Animal1_Movement/"
SAVE_DIR = r"D:\Data\Research\NEURO\movement\processed_data/"
PLOT_DIR = r"C:\Users\jackm\PycharmProjects\NEURO\NEURO_MOVEMENT\Data\plots/"

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


def create_neuron_limb_coord_data(data_dict, save_dir):
    base_filename = 'general_data'
    DATA = []

    limb_coord_raw = data_dict['behav_coord_likeli']
    intermediate_map = data_dict['idx_coord_neural']
    neuron_spks_raw = data_dict['spks_final']

    DATA.append(neuron_spks_raw)
    DATA.append(intermediate_map)
    DATA.append(limb_coord_raw)

    for i in range(len(DATA)):
        print(f"Data {i} shape: {DATA[i].shape}")
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


def create_neural_dataset(data_dict, save_dir, plot=False, standardize=False, delta=False, tokenize_neurons = False):
    neuron_spks_raw = data_dict['spks_final'].transpose()
    N, F = neuron_spks_raw.shape
    if delta:
        dneuron_spks_raw = np.empty(shape=(N, F))
        for i in range(N-1):
            dneuron_spks_raw[i] = neuron_spks_raw[i+1] - neuron_spks_raw[i]
        neuron_spks_raw = dneuron_spks_raw
        if standardize:
            sd = StandardScaler()
            neuron_spks_raw = sd.fit_transform(neuron_spks_raw)
            #neuron_spks_raw = (neuron_spks_raw - neuron_spks_raw.mean()) / neuron_spks_raw.std()
    if delta and standardize:
        neuron_spks_raw = np.cbrt(neuron_spks_raw)
    if plot:
        neuron_spks_raw_flattened = neuron_spks_raw.flatten()
        print(neuron_spks_raw_flattened.shape)
        plt.hist(neuron_spks_raw_flattened, bins=30)
        plt.yscale('log')
        plt.savefig(PLOT_DIR + 'neural_dist.png')
    if tokenize_neurons:
        neuron_spks_raw = np.around(neuron_spks_raw, decimals=0)
        uniques = np.unique(neuron_spks_raw)
        make_pos = np.min(uniques)
        for i in tqdm(range(N), desc="TOKENIZING"):
            for j in range(F):
                neuron_spks_raw[i][j] += np.abs(make_pos)
        uniques = np.unique(neuron_spks_raw)
        vocab = len(uniques)
        print(f"Vocab size: {vocab}")
        print("Vocab: ", uniques)
        neural_tensor = torch.from_numpy(neuron_spks_raw).type(torch.long)
        DATA = [neural_tensor, vocab]
        with open(save_dir + 'neural_dataset.pkl', 'wb') as f:
            pickle.dump(DATA, f)
        f.close()
    else:
        neural_tensor = torch.from_numpy(neuron_spks_raw).type(torch.long)
        print(neural_tensor.shape)
        with open(save_dir + 'neural_dataset.pkl', 'wb') as f:
            pickle.dump(neural_tensor, f)
        f.close()


def create_neu2limb_dataset(data_dict, save_dir, plot=False, standardize=False):
    limb_coord_raw = data_dict['behav_coord_likeli'].transpose()
    intermediate_map = data_dict['idx_coord_neural']
    neuron_spks_raw = data_dict['spks_final'].transpose()
    inputs = np.empty(shape=(9360, 3327))
    outputs = np.empty(shape=(9360, 8))
    for i in tqdm(range(len(neuron_spks_raw)-1)):
        inputs[i] = neuron_spks_raw[i+1] - neuron_spks_raw[i]
    for i in tqdm(range(len(neuron_spks_raw)-1)):
        first_occur = min(idx for idx, val in enumerate(intermediate_map) if val == i)
        last_occur = max(idx for idx, val in enumerate(intermediate_map) if val == i+1)
        outputs[i] = limb_coord_raw[last_occur] - limb_coord_raw[first_occur]
    if standardize:
        inputs = (inputs - inputs.mean()) / inputs.std()
        inputs = np.cbrt(inputs)
        outputs = (outputs - outputs.mean()) / outputs.std()
    if plot:
        inputs_flattened = inputs.flatten()
        outputs_flattened = outputs.flatten()
        bins = 40
        fig = plt.figure(figsize=(10, 8))
        plt.hist(inputs_flattened, bins=bins)
        plt.title('Neuron spk sample distribution')
        plt.yscale('log')
        plt.savefig(PLOT_DIR + 'neudist.png')
        plt.cla()
        plt.clf()
        fig = plt.figure(figsize=(10, 8))
        plt.hist(outputs_flattened, bins=bins)
        plt.title('limb coord sample distribution')
        plt.yscale('log')
        plt.savefig(PLOT_DIR + 'lcdist.png')
        plt.cla()
        plt.clf()
    inputs = torch.from_numpy(inputs)
    outputs = torch.from_numpy(outputs)
    DATA = [inputs, outputs]
    with open(save_dir + 'combined_data.pkl', 'wb') as f:
        pickle.dump(DATA, f)
    f.close()
    print("DATA SAVED :)")


if __name__ == '__main__':
    data_dict = load_npy_files(BASE_DIR)
    create_neuron_limb_coord_data(data_dict, SAVE_DIR)

