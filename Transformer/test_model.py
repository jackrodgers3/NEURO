import torch
import numpy as np
from encmodel import EncModel

model = EncModel(input_size=1, target_size=1719, N=8, d_model=512, d_ff=2048,
                 h=8, dropout=0.1, bias=True)

#model = EncModel()
model.load_state_dict(torch.load(r"C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023\Output\enc_model.pt"))
model.eval()
test_input = torch.tensor([1]).type(torch.FloatTensor)
result = model(test_input)
avg = 0.0
for i in range(1719):
    avg += abs(result[0][i].item())
avg /= float(1719)

for i in range(1719):
    if abs(result[0][i].item()) < 0.01:
        result[0][i] = 0.0
    else:
        print(result[0][i].item())
print(result)