# Randomized Gradient Boosting Machine (RGBM)

This repo contains demo implementations of Randomized Gradient Boosting Machine based on the following paper:

> Haihao Lu and Rahul Mazumder. _Randomized Gradient Boosting Machine._ https://arxiv.org/abs/1810.10158

# Usage

## Train the model
Example: Train for 100 epoches of a1a by RGBM with different t value (size of the features chosen in RGBM).
```bash
python libsvm_data.py 100 a1a
```
The code prints the final iteration training loss and testing loss for RGBM, store the outputs at `../output/a1a.mat`.

To reproduce the figures appeared in the paper, run
```bash
python libsvm_data.py 200 a9a
python libsvm_data.py 200 colon-cancer
python libsvm_data.py 20 rcv1
python libsvm_data.py 200 YearPredictionMSD_t
python plot_figures_from_output.py
```
(Warning: rcv1 is a big dataset, and it can take a couple of hours to finish the run.)

