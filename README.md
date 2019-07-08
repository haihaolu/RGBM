# Randomized Gradient Boosting Machine

This repo contains demo implementations of the a1a training code based on the following paper:

> Haihao Lu and Rahul Mazumder. _Randomized Gradient Boosting Machine._ https://arxiv.org/abs/1810.10158

## Trains the model
```bash
# train for 100 epoches of a1a by RGBM with different t value, store the outputs at `../output/a1a.mat` and plots the figures at `../figures/`
python code/libsvm_data.py 100 a1a
```

