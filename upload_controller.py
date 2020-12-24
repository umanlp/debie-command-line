from controller import calculation
import numpy as np
import io
from data_controller import load_binary_embeddings

uploaded_binary = ''

uploaded_filename = ''
uploaded_space = {}

uploaded_vectorfile = ''
uploaded_vocabfile = ''
uploaded_vecs = []
uploaded_vocab = {}


def get_vocab_vecs_from_upload():
    if uploaded_binary is 'true':
        return uploaded_vocab, uploaded_vecs
    else:
        return calculation.dict_to_vocab_vecs(uploaded_space)


def load_dict_uploaded_file(filename):
    global uploaded_filename
    global uploaded_space
    path = filename
    fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        ln = np.array(tokens[1:])
        data[tokens[0]] = ln.astype(np.float)
    uploaded_filename = filename
    uploaded_space = data
    return data


def load_binary_uploads(vocab_filename, vecs_filename):
    path_vocab = vocab_filename
    path_vecs = vecs_filename
    vocab, vecs = load_binary_embeddings(path_vocab, path_vecs)
    return vocab, vecs
