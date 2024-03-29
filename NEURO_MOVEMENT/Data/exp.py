import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn

SAVE_DIR = r"C:\Users\jackm\PycharmProjects\NEURO\NEURO_MOVEMENT\Data\plots/"
data = np.array([1, 5, 2, 3, 0, 0, 2, 3])
c = np.unique(data)
print(c)
b = torch.tensor(data, dtype=torch.int32)
emb = nn.Embedding(6, 10)
out = emb(b)
print(out)