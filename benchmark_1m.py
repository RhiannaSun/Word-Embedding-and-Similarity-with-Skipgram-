import itertools
import nltk
import string

import numpy as np
import pandas as pd

from collections import Counter
from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
nltk.download('wordnet')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuations = string.punctuation

# ps = PorterStemmer()
tags_ignore = {'punct', 'cc'}
conll_punc = {'-lrb-': '(', '-rrb-': ')', '-backslash-': '\\'}

max_data_size = 10000

def is_valid_word(word):
    return word not in stop_words and word.isalpha() and len(word) > 1

def write_vocab(data, f_name):
    vocab = {word for tup in data for word in tup}
    vocab = [[w, idx] for idx, w in enumerate(vocab)]
    df = pd.DataFrame(vocab, columns=['word', 'index'])
    df.to_csv(f_name, index=False)

def sub_weight(data, threshold=1e-5):
    word_list = list(itertools.chain(*data))
    threshold = threshold * len(word_list)
    print(f'Word List: {len(word_list)}')
    drop_prob = {word: 1 - np.sqrt(threshold/count) for word, count in Counter(word_list).items()}
    return drop_prob


def sub_high(data, max_threshold=2000, min_threshold=5):
    print(f'Original size: {len(data)}')
    data = np.array(data)
    np.random.shuffle(data)

    counter = Counter(data[:, 1])
    # find the terms with counts higher than threshold
    frequent_words = {w: 0 for w, count in counter.items() if count > max_threshold}

    subsampled = [(x, y) for x, y in data if y not in frequent_words and counter[y] > min_threshold]
    for x, y in data:
        if y in frequent_words and frequent_words[y] < max_threshold:
            
            frequent_words[y] += 1
            subsampled.append((x, y))

    print(f'Size after subsampling: {len(subsampled)}')
    return subsampled


def get_neighbors(words, n):
    # Adding start and end tokens
    words = ['A'] + words + ['Z']
    word_pairs = []
    for i in range(1, len(words)-1):
        neighbors = words[max(0, i-n): i] + words[i+1: min(len(words), i+n)+1]
        data_points = [(ctx, words[i]) for ctx in neighbors]
        word_pairs.extend(data_points)
    return word_pairs


def get_1m_conll_data(f_name='data/training/training-data.1m.conll', max_data_size=100000, min_count=10):
    conll_labels, word_map = [], {}
    print(f'Loading CONLL dependency data')
    with open(f_name, 'r') as f:
        for line in f.readlines()[:max_data_size]:
            labels = line.lower().strip().split('\t')

            if len(labels) > 1:
                if labels[7] not in tags_ignore:
                    word_map[int(labels[0])] = (stemmer.stem(labels[1]), labels[7], int(labels[6]))

            elif len(word_map) > 0:
                for _, (word, lbl, head_idx) in word_map.items():
                    if lbl not in {'ROOT', 'prep'} and head_idx in word_map:
                        if lbl == 'pobj' and word_map[head_idx][1] == 'prep':
                            head = word_map[word_map[head_idx][2]]
                            lbl = f'prep_{word_map[head_idx][0]}'
                        else:
                            head = word_map[head_idx][0]
                        conll_labels.append((f'{word}_{lbl}', head))
                        conll_labels.append((f'{head}_{lbl}_Z', word))

                word_map = {}
            else:
                word_map = {}
    print(f'Finished loading CONLL data, Size={len(conll_labels)}')

    return sub_high(conll_labels, max_threshold=2000)


def get_1m_bow_data(n=4, subsampling='sub-high', high_thresh=2000, weigt_thresh=1e-5, max_size=30000):
    with open('data/training/training-data.1m', 'r') as f:
        data = [[stemmer.stem(word) for word in line.lower().split() if is_valid_word(word)] for line in f.readlines()[:max_size]]
    print('Training Data loaded')

    if subsampling == 'sub-high':
        train_data = []
        for data_i in data:
            train_data.extend(get_neighbors(data_i, n))
        train_data = sub_high(train_data, max_threshold=high_thresh)

    elif subsampling == 'sub-weight':
        drop_prob = sub_weight(data, threshold=weigt_thresh)
        train_data = []
        for data_i in data:
            rands = np.random.rand(len(data_i))
            subsampled = [word for idx, word in enumerate(data_i) if rands[idx] > drop_prob[word]]
            train_data.extend(get_neighbors(subsampled, n))
    
    write_vocab(train_data, 'data/vocab.csv')
    print(f'Training Data Pre-processed, Size={len(train_data)}')
    return train_data

