# CS5740 A2

In  this  assignment,  we  present  results  from  our experiment of creating vector representations for English words, such that these word embeddings accurately represent the syntactic and semantic  similarity  between  word pairs. A  subset  of  the  [1 Billion Word Language Model Benchmark](https://www.statmt.org/lm-benchmark/) corpus is used to train the skip-gram model for these experiments, which involve both, the raw text and the POS-tagged corpus.

Spearman's rank correlation is used to measure the similarity between the embeddings predicted by the model and that computed by humans.


## Execution Instructions

To setup the environment for this project, use Python 3.8.x and install all requirements.
```
pip install -r requirements.txt
```

Run the following command to train the model. This will train the data on 500,000 sentences.
```
python network.py
```

This will generate an `embeddings.txt` which contains word embeddings for words occuring in the development and test set. Their cosine similarity and Spearman's rank can be computed as follows:
```
python similarity.py > results/dev_scores.csv -e data/embeddings.txt -w data/similarity/dev_x.csv
python evaluate.py -p results/dev_scores.csv -d data/similarity/dev_y.csv
```

## Results

The results reported on the development and test set from these experiments are:

| Dataset Size | 200k | 500k | 800k |
| ----------- | ----------- | ---- | ----- |
| Development Set | 0.539 | 0.547 | 0.513|
| Test Set | 0.204 | 0.2803 | N/A |
