# Reweighted Wake-Sleep for Probabilistic Programs (a.k.a. Coordinate Descent Auto-Encoder)

## GMM Open Universe

In `code/gmm-open-universe`, run

```
python gmm_open_universe_vae.py
python gmm_open_universe_vae_relax.py
python gmm_open_universe_cdae.py
```

to train the generative and inference networks. Then run

```
python plot.py
```

to obtain plots.

## MNIST/Omniglot

In `code/mnist-omniglot`, run

```
python train.py --help
```

to see instructions to train the generative and inference networks. Then run

```
python plot.py
```

to obtain plots.

## Hidden Markov Model

Dependencies:
- https://github.com/hmmlearn/hmmlearn
- https://github.com/tuananhle7/aesmc/tree/master/code

In `code/hmm`, run

```
python train.py --help
```

to see instructions to train the generative and inference networks. Then run

```
python plot.py
```

to obtain plots.
