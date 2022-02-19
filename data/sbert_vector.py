from sklearn.datasets import fetch_20newsgroups
import re
import string
import numpy as np
import joblib
import torch
from sentence_transformers import SentenceTransformer

train_data = fetch_20newsgroups(subset='train')
test_data = fetch_20newsgroups(subset='test')
init_docs_tr = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', train_data.data[doc]) for doc in range(len(train_data.data))]
init_docs_ts = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', test_data.data[doc]) for doc in range(len(test_data.data))]


def contains_punctuation(w):
    return any(char in string.punctuation for char in w)


def contains_numeric(w):
    return any(char.isdigit() for char in w)


doc_sentences = init_docs_tr + init_docs_ts
doc_sentences = [[w.lower() for w in doc_sentences[doc] if not contains_punctuation(w)] for doc in range(len(doc_sentences))]
doc_sentences = [[w for w in doc_sentences[doc] if not contains_numeric(w)] for doc in range(len(doc_sentences))]
doc_sentences = [[w for w in doc_sentences[doc] if len(w) > 1] for doc in range(len(doc_sentences))]
doc_sentences = [" ".join(doc_sentences[doc]) for doc in range(len(doc_sentences))]
all_label = np.concatenate((train_data.target, test_data.target), axis=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
doc_embeddings = model.encode(sentences=doc_sentences, batch_size=32, show_progress_bar=True, convert_to_numpy=True, device=device)

newsgroups_data = {'doc_sentences': doc_sentences, 'doc_embeddings': doc_embeddings, 'all_label': all_label, 'train_len': len(train_data.target), 'test_len': len(test_data.target)}
joblib.dump(newsgroups_data, "20ng/data_20ng_SBERT_embedding.pkl")

