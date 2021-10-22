import pickle
import bz2
import nltk

import pandas as pd
import numpy as np

from collections import Counter

from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
nltk.download('wordnet')

def train_word_distribution():
    words_train = []
    with open('data/training/training-data.1m', 'r') as f:
        for line in f.readlines():
            ws = [stemmer.stem(w) for w in line.strip().lower().split()]
            words_train.extend(ws)

    words_train_counter = Counter(words_train)
    print(f'Most Common 300: {words_train_counter.most_common(300)}')

def create_rare_words():
    rare = [w for w, c in get_train_counter().items() if c < 20]
    np.savetxt('data/rare_words.txt', np.array(rare), fmt='%s')

def find_synonym(word):
    return {stemmer.stem(l.name()).lower() for syn in wordnet.synsets(word) for l in syn.lemmas()}

def find_char_ngrams(word, n=3):
    return {word[i: i + n_i]
              for n_i in range(n, len(word)) for i in range(len(word) + 1 - n_i)}

def get_words_from_file(dev_or_test='dev'):
    words = set()
    word_pairs = pd.read_csv(f'data/similarity/{dev_or_test}_x.csv', index_col = "id")
    for w1, w2 in zip(word_pairs.word1, word_pairs.word2):
        words.update([w1, w2])
    return words

def get_train_counter(fp='data/training/training-data.1m'):
    words_train = []
    with open(fp, 'r') as f:
        for line in f.readlines():
            w = [stemmer.stem(l) for l in line.strip().lower().split()]
            words_train.extend(w)
    return Counter(words_train)

def write_to_pickle(f_name, data, compress=False):
    f = bz2.BZ2File(f_name, 'w') if compress else open(f_name, 'wb')
    pickle.dump(data, f)
    f.close()

def read_pickle(f_name, is_compressed=False):
    f = bz2.BZ2File(f_name, 'r') if is_compressed else open(f_name, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def write_conll_1m():
    from benchmark_1m import get_1m_conll_data
    data = get_1m_conll_data(min_count=10)
    write_to_pickle('data/pickled/conll_1m.pkl', data, compress=True)

def write_bow_1m():
    from benchmark_1m import get_1m_bow_data
    data = get_1m_bow_data(subsampling='sub-high',max_size=300000)
    # import pdb; pdb.set_trace()
    X = pd.DataFrame(data)
    X.to_csv('data_bow_300k.csv',index=False)

def read_bow_1m():
    return read_pickle('data/pickled/bow_1m.pkl', is_compressed=True)


def create_embedding_file(weights, vocab_words, word_counter):
    embedding_map = dict(zip(vocab_words, weights))
    
    vocab_words = set(vocab_words)
    low_freq_words = np.loadtxt('data/rare_words.txt', dtype='str').tolist()
    avg_missing_weight = np.mean((np.array([embedding_map[word] for word in low_freq_words if word in vocab_words])), axis=0)

    dev_test_words = get_words_from_file(dev_or_test='dev') | get_words_from_file(dev_or_test='test')
    embeddings = [None] * len(dev_test_words)
    in_train, in_syn, in_char_ngram, missing = 0, 0, 0, 0
    word_miss = []

    for idx, word in enumerate(dev_test_words):
        stem_word = stemmer.stem(word).lower()
        if stem_word in vocab_words and stem_word not in low_freq_words:
            curr_wt = embedding_map[stem_word]
            in_train += 1
        else:
            synonyms = ({stem_word} | find_synonym(word)) & vocab_words
            if len(synonyms) > 0: # Set embeddings to be average of all synonyms
                curr_wt = np.mean((np.array([embedding_map[word] for word in synonyms])), axis=0)
                in_syn += 1
            else:
                char_ngram = find_char_ngrams(word) & vocab_words
                char_ngram = set(filter(lambda x: [x for i in char_ngram if x in i and x != i] == [], char_ngram))
                if len(char_ngram) > 0: # Set embeddings to be average of all hypernyms
                    curr_wt = np.mean((np.array([embedding_map[word] for word in char_ngram])), axis=0)
                    in_char_ngram += 1
                else:
                    missing += 1
                    word_miss.append(word)
                    curr_wt = avg_missing_weight
        
        embeddings[idx] = [word] + curr_wt.tolist()
    
    print(f"(In Training, In Synonyms, Char-Ngram, Missing) = {(in_train, in_syn, in_char_ngram, missing)}")
    print(f'Missing Words: {word_miss}')
    np.savetxt('data/embeddings.txt', np.array(embeddings), fmt='%s')

if __name__ == '__main__':
    print('This is utils.py')
