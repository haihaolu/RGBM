# Randomized Gradient Boosting Machine (RGBM)

This repo contains demo implementations of Randomized Gradient Boosting Machine based on the following paper:

> Haihao Lu and Rahul Mazumder. _Randomized Gradient Boosting Machine._ https://arxiv.org/abs/1810.10158

# Usage

## Train the model
Example: Train for 100 epoches of a1a by RGBM with different t value (size of the features chosen in RGBM).
```bash
python libsvm_data.py 100 a1a
```
The code prints the final iteration training loss and testing loss for RGBM with different t values, store the outputs at `../output/a1a.mat`, and plot figures at `../figures/`.

To reproduce the figures appeared in the paper, run
```bash
python solve_problems.py 200 a9a
python solve_problems.py 200 colon-cancer
python solve_problems.py 20 rcv1
python solve_problems.py 200 YearPredictionMSD_t
python plot_figures_from_output.py
```
(Warning: The above code can take a couple of hours to finish the run.)

This code has been tested with Python 2.7.14.

