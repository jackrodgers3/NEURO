import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

###HYPERPARAMETERS + GLOBAL DATA######
BASE_DIR = r'C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023/Data/'
BASE_DIR2 = r'C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023/Transformer/'
D_DRIVE = r'D:\Data\Research\NEURO\touch/'
path = BASE_DIR + 'Animal2_Touch'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################

#preprocessing
# behavior --> idx_coord_neural --> neural data
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

data_set = load_npy_files(path)
limb_order = []
for i in range(len(data_set['touch_behav'])):
    limb_order.append(data_set['touch_behav'][i][2])

coord_to_spk = []
for j in range(len(data_set['idx_coord_neural'])):
    coord_to_spk.append(data_set['spks_final'].transpose()[data_set['idx_coord_neural'][j]])

limb_targets = []
count = 0
for k in range(len(coord_to_spk)):
    if data_set['touch_behav'][count][0] > k:
        limb_targets.append(0)
    elif data_set['touch_behav'][count][0] <= k <= data_set['touch_behav'][count][1]:
        limb_targets.append(limb_order[count])
    else:
        limb_targets.append(0)
        count +=1
inputs = torch.from_numpy(np.array(coord_to_spk))
targets = torch.from_numpy(np.array(limb_targets))

#using transformer-like model to try to predict limb# -> neuron signals

inputs = torch.from_numpy(np.array(inputs)).type(torch.FloatTensor)
targets = torch.from_numpy(np.array(targets)).type(torch.FloatTensor)
print(f'Neuron signals from each frame shape: {inputs.shape}')
print(f'Limb moved in each frame shape: {targets.shape}')
DATA = []
DATA.append(inputs)
DATA.append(targets)
with open(D_DRIVE + 'neuron_to_limb2.pkl', 'wb') as f:
    pickle.dump(DATA, f)
f.close()

print(type(inputs[0][0].item()))




def process_data(base_dir):
    subfolders = os.listdir(base_dir)
    inputs = []
    targets = []
    for i in range(len(subfolders)):
        data_set = load_npy_files(base_dir + subfolders[i])
        limb_order = []
        for i in range(len(data_set['touch_behav'])):
            limb_order.append(data_set['touch_behav'][i][2])

        coord_to_spk = []
        for j in range(len(data_set['idx_coord_neural'])):
            coord_to_spk.append(data_set['spks_final'].transpose()[data_set['idx_coord_neural'][j]])

        limb_targets = []
        count = 0
        for k in range(len(coord_to_spk)):
            if data_set['touch_behav'][count][0] > k:
                limb_targets.append(0)
            elif data_set['touch_behav'][count][0] <= k <= data_set['touch_behav'][count][1]:
                limb_targets.append(limb_order[count])
            else:
                limb_targets.append(0)
                count += 1
        inputs.append(np.array(coord_to_spk))
        targets.append(np.array(limb_targets))
    print("Frame format: 30Hz")
    print(f'Neuron signals from each frame shape: {len(inputs)}')
    print(f'Limb moved in each frame shape: {len(targets)}')
    return inputs, targets





