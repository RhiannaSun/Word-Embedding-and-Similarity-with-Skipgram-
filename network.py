import warnings
warnings.filterwarnings('ignore')

import csv
import sys
import time
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

from collections import Counter
from benchmark_1m import get_1m_conll_data, get_1m_bow_data
from utils import create_embedding_file

class MyNetwork(nn.Module):
    def __init__(self, vocab_ctx, vocab_word, embed_dim=128):
        super(MyNetwork, self).__init__()
        self.context_fn = nn.Embedding(vocab_ctx, embed_dim)
        self.word_fn = nn.Embedding(vocab_word, embed_dim)

        nn.init.xavier_normal(self.context_fn.weight)
        nn.init.xavier_normal(self.word_fn.weight)

    def forward(self, ctx, word):
        ctx_vec = self.context_fn(ctx)
        word_vec = self.word_fn(word)
        return torch.mul(ctx_vec, word_vec).sum(1, keepdim=True)

def create_vocab(data, min_count=10):
    return {v: idx for idx, v in enumerate(set(data))}

def generate_neg_samples(data, num_neg):
    data = set(data)
    data_x, data_y = list(zip(*data))
    # Context Unigram Distribution
    context_dist_keys, context_dist = zip(*Counter(data_x).most_common())
    context_dist = np.array(context_dist)**0.75
    context_dist = context_dist/context_dist.sum()

    context_neg = np.random.choice(context_dist_keys, size=(len(data_y), num_neg), p=context_dist)
    return [(ctx, w) for idx, w in enumerate(data_y) for ctx in context_neg[idx] if (ctx, w) not in data]

def split_train_dev(data, train_ratio=0.95):
    split_point = round(train_ratio * len(data))
    train_set, dev_set = data[:split_point], data[split_point:]
    return train_set, dev_set

def train(epochs=20, neg_sample_rate=5):
    has_cuda = torch.cuda.is_available()
    print(f'Is CUDA available: {has_cuda}')
    data_x, data_y = map(list, zip(*get_1m_bow_data(max_size=500000)))

    vocab = {}
    with open('data/vocab.csv', mode='r') as f:
      csv_reader = csv.reader(f)
      _ = next(csv_reader)
      vocab = {row[0]: int(row[1]) for row in csv_reader}
    
    # Preparing the dataset
    pos_samples = [(vocab[x], vocab[y]) for x, y in zip(data_x, data_y)]

    # Generate negative samples
    print(f'Generating Negative Samples: {neg_sample_rate} per positive sample')
    neg_samples = generate_neg_samples(pos_samples, neg_sample_rate)

    print(f'Shuffling the dataset')
    data = np.array(list(zip(pos_samples, np.ones(len(pos_samples)))) + list(zip(neg_samples, np.zeros(len(neg_samples)))))
    data_indices = np.arange(data.shape[0])
    np.random.shuffle(data_indices)

    train_set, dev_set = split_train_dev(data[data_indices.tolist()])
    print(f'Train: {len(train_set)} Dev:{len(dev_set)}, Vocab Size: {len(vocab)}')
    word_contexts, labels = zip(*train_set)
    data_x, data_y = zip(*word_contexts)

    # Create model
    model = MyNetwork(len(vocab), len(vocab), embed_dim=300)
    if has_cuda:
        model.cuda()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 8192
    print(f'---Start training model---')
    for ep in range(epochs):
        train_idx = np.arange(train_set.shape[0])
        np.random.shuffle(train_idx)
        train_set = train_set[train_idx.tolist()]
        
        for i in range(0, len(train_set), batch_size):
            end = min(i+batch_size, len(train_set))
            
            lbl = torch.tensor(labels[i: end], dtype=torch.float)
            x = torch.tensor(data_x[i: end], dtype=torch.long)
            y = torch.tensor(data_y[i: end], dtype=torch.long)

            if has_cuda:
                lbl, x, y = lbl.cuda(), x.cuda(), y.cuda()
            
            # Forward Pass
            dot_prod = model(x, y)

            # Compute and print loss
            loss = loss_fn(dot_prod, lbl.view(dot_prod.shape[0], 1))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if ep % 2 == 0:
            print(f'Epoch={ep}, Loss={loss.data}, Time={time.time()}')

    model.eval()
    # Dev Loss
    dev_word_contexts, dev_labels = zip(*dev_set)
    dev_x, dev_y = zip(*dev_word_contexts)

    loss = []
    for i in range(0, len(dev_set), batch_size):
        end = min(i+batch_size, len(dev_set))

        lbl = torch.tensor(dev_labels[i: end], dtype=torch.float)
        x = torch.tensor(dev_x[i: end], dtype=torch.long)
        y = torch.tensor(dev_y[i: end], dtype=torch.long)

        if has_cuda:
            x, y, lbl = x.cuda(), y.cuda(), lbl.cuda()
        dot_prod = model(x, y)
        loss.append(loss_fn(dot_prod, lbl.view(dot_prod.shape[0], 1)))
    print(f'Dev Loss: {sum(loss)/len(loss)}')

    # Write Embeddings to file
    weights = model.word_fn.weight.detach().cpu().numpy() if has_cuda else model.word_fn.weight.detach().numpy()
    vocab_words = sorted(vocab, key=vocab.get)
    create_embedding_file(weights, vocab_words, Counter(data_y))


if __name__ == '__main__':
    train(epochs=20, neg_sample_rate=5)