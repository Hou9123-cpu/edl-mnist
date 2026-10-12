"""Generate the figures used in the report."""
import os

import matplotlib
matplotlib.use("Agg")  # headless backend, no display required
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, N_CLASSES


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("Confusion matrix (test set)")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(per_class, path):
    classes = sorted(per_class.keys(), key=int)
    f1 = [per_class[c]["f1"] for c in classes]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(classes, f1, color="#4c72b0")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Digit class")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 score (test set)")
    for b, v in zip(bars, f1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(history, path):
    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("EDL loss")
    ax1.set_title("Training / validation loss")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"], color="#c44e52")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_epistemic_uncertainty(epistemic, correct_mask, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(epistemic[correct_mask], bins=40, alpha=0.6,
            label="Correct", color="#55a868")
    ax.hist(epistemic[~correct_mask], bins=40, alpha=0.6,
            label="Incorrect", color="#c44e52")
    ax.set_xlabel("Epistemic uncertainty (nats)")
    ax.set_ylabel("Count")
    ax.set_title("Epistemic uncertainty: correct vs. incorrect predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
