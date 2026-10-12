"""MNIST subset loading with a fixed random seed and no test-set leakage.

The official MNIST train/test split is preserved: training and validation
examples come from ``train=True``, and the test examples come from
``train=False``, so no test image ever appears in training or validation.
"""
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from config import SEED, N_TRAIN, N_VAL, N_TEST, DATA_DIR


def build_subsets():
    """Return (train_subset, val_subset, test_subset) deterministically.

    The training/validation split uses a ``torch.Generator`` seeded with
    ``SEED`` so the exact same 2,000 / 500 images are selected every run.
    """
    transform = transforms.Compose([transforms.ToTensor()])  # scales pixels to [0, 1]
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_full = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    gen = torch.Generator().manual_seed(SEED)
    train_idx = torch.randperm(len(train_full), generator=gen).tolist()

    tr_idx = train_idx[:N_TRAIN]
    va_idx = train_idx[N_TRAIN:N_TRAIN + N_VAL]
    # Official test set, first N_TEST examples in natural order (labels preserved).
    te_idx = list(range(N_TEST))

    return (Subset(train_full, tr_idx),
            Subset(train_full, va_idx),
            Subset(test_full, te_idx))
