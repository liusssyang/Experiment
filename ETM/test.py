import os

import torch

from ETM import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('results/etm_20ng_K_50_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_1', 'rb') as f:
    model = torch.load(f)
model = model.to(device)
model.eval()

data_path = '../data/20ng'
vocab, train, valid, test = data.get_data(os.path.join(data_path))
vocab_size = len(vocab)
print()
