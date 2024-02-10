import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pickle
base_dir = r"C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023\Transformer/"
#make delta dataset

with open(base_dir + "inputs_pickle", "rb") as f:
    inputs = pickle.load(f)
f.close()
print(inputs.shape)
sys.exit()

delta_tensor = torch.empty(size=(45358, 901))
for i in range(35357):
    delta_tensor[i] = inputs[i+1] - inputs[i]


with open(base_dir + "delta_inputs2_pickle", "wb") as g:
    pickle.dump(delta_tensor, g)
g.close()
