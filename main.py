import torch
import random
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
max_df = 0.7
min_df = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sbert_result = load(r"data/20ng/data_20ng_SBERT_embedding.pkl")
etm_theta_result = load(r"data/20ng/data_20ng_etm_theta.pkl")
# print(sbert_result.keys())
sentence = sbert_result['doc_sentences']
sentences_vecs = sbert_result["doc_embeddings"]
sentence_label = sbert_result["all_label"]
doc_theta = etm_theta_result['theta']


def take_sample(sample_num):
    sbert_doc_embeddings_samples = []
    etm_theta_vectors_samples=[]
    _20ng_label = []

    my_random = random.sample(range(0, len(sentences_vecs) + 1), sample_num)
    for i in range(0, sample_num):
        sbert_doc_embeddings_samples.append(sentences_vecs[my_random[i]])
        etm_theta_vectors_samples.append(doc_theta[my_random[i]])
        _20ng_label.append(sentence_label[my_random[i]])
    cos_similarity_sbert = cosine_similarity(sbert_doc_embeddings_samples)
    cos_similarity_etm_theta=cosine_similarity(etm_theta_vectors_samples)
    return cos_similarity_sbert, cos_similarity_etm_theta,_20ng_label, my_random


take_sample(sample_num=10)
print()
