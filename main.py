import os

import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import random
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import itertools
from scipy.io import savemat, loadmat
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pickle

# Maximum / minimum document frequency
max_df = 0.7
min_df = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = load(r"data/20ng/data_20ng_SBERT.pkl")
print(data.keys())
sentence = data['doc_sentences']
sentences_vecs = data["doc_embeddings"]
sentence_label = data["all_label"]


def take_sample(sample_num):
    _ = []
    lable = []
    my_random = random.sample(range(0, len(sentences_vecs) + 1), sample_num)
    for i in range(0, sample_num):
        _.append(sentences_vecs[my_random[i]])
        lable.append(sentence_label[my_random[i]])
    cos = cosine_similarity(_)
    return cos, lable, my_random


take_sample(sample_num=100)
print()
