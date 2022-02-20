from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from scipy import sparse
import re
import string
import joblib

# Maximum / minimum document frequency
max_df = 0.7
min_df = 10  # choose desired value for min_df

# Read stopwords
with open(r'C:\Users\LY\PycharmProjects\Experiment\data\stops.txt', 'r') as f:
    stops = f.read().split('\n')

train_data = fetch_20newsgroups(subset='train')
test_data = fetch_20newsgroups(subset='test')
init_docs_tr = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', train_data.data[doc]) for doc in range(len(train_data.data))]
init_docs_ts = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', test_data.data[doc]) for doc in range(len(test_data.data))]


def contains_punctuation(w):
    return any(char in string.punctuation for char in w)


def contains_numeric(w):
    return any(char.isdigit() for char in w)


def remove_empty(in_docs):
    return [doc for doc in in_docs if doc != []]


def create_list_words(in_docs):
    return [x for y in in_docs for x in y]


def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]


def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1] * len(doc_indices), (doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()


def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc, :].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc, :].data] for doc in range(n_docs)]
    return indices, counts


init_docs = init_docs_tr + init_docs_ts
init_docs = [[w.lower() for w in init_docs[doc] if not contains_punctuation(w)] for doc in range(len(init_docs))]
init_docs = [[w for w in init_docs[doc] if not contains_numeric(w)] for doc in range(len(init_docs))]
init_docs = [[w for w in init_docs[doc] if len(w) > 1] for doc in range(len(init_docs))]
init_docs = [" ".join(init_docs[doc]) for doc in range(len(init_docs))]
# Create count vectorizer

cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
cvz = cvectorizer.fit_transform(init_docs).sign()
# Get vocabulary

sum_counts = cvz.sum(axis=0)
v_size = sum_counts.shape[1]
sum_counts_np = np.zeros(v_size, dtype=int)
for v in range(v_size):
    sum_counts_np[v] = sum_counts[0, v]
word2id = dict([(w, cvectorizer.vocabulary_.get(w)) for w in cvectorizer.vocabulary_])
id2word = dict([(cvectorizer.vocabulary_.get(w), w) for w in cvectorizer.vocabulary_])

# Sort elements in vocabulary
idx_sort = np.argsort(sum_counts_np)
vocab_aux = [id2word[idx_sort[cc]] for cc in range(v_size)]
# Filter out stopwords (if any)
vocab_aux = [w for w in vocab_aux if w not in stops]

# Create dictionary and inverse dictionary
vocab = vocab_aux
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])
# Split in train/test/valid

num_docs_tr = len(init_docs_tr)
trSize = num_docs_tr - 100
tsSize = len(init_docs_ts)
vaSize = 100
allSize = len(init_docs)
idx_permute = np.random.permutation(num_docs_tr).astype(int)
# Remove words not in train_data
vocab = list(set([w for idx_d in range(trSize) for w in init_docs[idx_permute[idx_d]].split() if w in word2id]))
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])

docs_all = [[word2id[w] for w in init_docs[idx_d].split() if w in word2id] for idx_d in range(allSize)]
docs_all = remove_empty(docs_all)
words_all = create_list_words(docs_all)
doc_indices_all = create_doc_indices(docs_all)
n_docs_all = len(docs_all)
bow_tr = create_bow(doc_indices_all, words_all, n_docs_all, len(vocab))

bow_tokens, bow_counts = split_bow(bow_tr, n_docs_all)
etm_vector_data = {'bow_tokens': bow_tokens, 'bow_counts': bow_counts, "train_len": trSize, "valid_len": vaSize, "test_len": tsSize}
joblib.dump(etm_vector_data, r"C:\Users\LY\PycharmProjects\Experiment\data\20ng\etm_enc_input.pkl")
