import torch
from scipy.io import loadmat
import joblib
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_file = './result/etm_20ng_K_50_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_1'
token_file = '../data/20ng/bow_all_tokens.mat'
count_file = '../data/20ng/bow_all_counts.mat'
with open(model_file, 'rb') as f:
    model = torch.load(f, map_location=device)
model = model.to(device)
vocab = joblib.load("../data/20ng/vocab.pkl")
tokens = loadmat(token_file)['bow_tokens'].squeeze()
counts = loadmat(count_file)['bow_counts'].squeeze()
vocab_size = len(vocab)
num_docs = len(tokens)
data_batch = np.zeros((num_docs, vocab_size))
for doc_id in range(0, num_docs):
    doc = tokens[doc_id]
    count = counts[doc_id]
    if len(doc) == 1:
        doc = [doc.squeeze()]
        count = [count.squeeze()]
    else:
        doc = doc.squeeze()
        count = count.squeeze()
    for j, word in enumerate(doc):
        data_batch[doc_id, word] = count[j]
data_batch = torch.from_numpy(data_batch).float().to(device)
sums = data_batch.sum(1).unsqueeze(1)
normalized_data_batch = data_batch / sums
theta, _ = model.get_theta(normalized_data_batch)
data = {'theta': theta.detach().cpu().numpy()}
joblib.dump(data, '../data/20ng/data_20ng_etm_theta.pkl')
print("Have saved trained ETM-theta!")
