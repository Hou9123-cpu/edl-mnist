"""Experiment configuration for the Evidential Deep Learning (EDL) MNIST experiment.

All hyperparameters and paths are centralized here so the exact configuration
can be recorded to ``results/config.json`` and reproduced.
"""
import os

# Reproducibility
SEED = 42

# Dataset subset sizes. The test subset is taken from the official MNIST test
# split (labels preserved, natural order), so it is guaranteed disjoint from
# the training and validation subsets.
N_TRAIN = 2000
N_VAL = 500
N_TEST = 1000

# Model / training
N_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
ANNEALING_STEPS = 10  # KL regularizer weight ramps to 1.0 by this epoch

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)
