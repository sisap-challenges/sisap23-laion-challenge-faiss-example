# SISAP 2023 LAION2B Challenge: Example using faiss

This repository gives a small working example of solving tasks in the [SISAP 23 LAION2B Indexing Challenge](https://sisap-challenges.github.io/).

## Requirements

This code was tested using Python 3.8 using Anaconda. 
See <https://github.com/sisap-challenges/sisap23-laion-challenge-faiss-example/blob/master/.github/workflows/ci.yml> for an example installation of FAISS in this setup. 

## Running

Running
```
$ python search/search.py
```

takes care of 

- downloading datasets/query sets used to benchmark the index to `data/`
- run 30-NN queries on the index for each query in the query set using a couple of different hyperparameters,
- stores results in `result/` as hdf5 files that can be post-processed using <https://github.com/sisap-challenges/sisap23-laion-challenge-evaluation>. This repository is added as a submodule. 

## Evaluation

```
$ python eval/eval.py
```

