import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

with open(r'C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023\Transformer\inputs2_pickle', 'rb') as f:
    INPUTS = pickle.load(f)
f.close()

with open(r'C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023\Transformer\targets2_pickle', 'rb') as g:
    TARGETS = pickle.load(g)
g.close()

