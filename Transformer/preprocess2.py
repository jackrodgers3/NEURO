import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
D_DRIVE = r'D:\Data\Research\NEURO\touch/'
import pickle
base_dir = r"C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023\Transformer/"
#make delta dataset

with open(D_DRIVE + "neuron_to_limb2.pkl", "rb") as f:
    data = pickle.load(f)
f.close()
inputs, outputs = data
print(inputs.shape)
inp_l = len(inputs)


delta_tensor = torch.empty(size=(inp_l-1, 901))
for i in range(inp_l-1):
    delta_tensor[i] = inputs[i+1] - inputs[i]
outputs = outputs[1:]
NEWDATA = [delta_tensor, outputs]
print(delta_tensor.shape)
print(outputs.shape)
with open(D_DRIVE + "dneuron_to_limb2.pkl", "wb") as g:
    pickle.dump(NEWDATA, g)
g.close()
