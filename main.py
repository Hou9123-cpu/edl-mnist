"""Entry point for the Evidential Deep Learning MNIST experiment.

Usage:
    python main.py            # run the full experiment
    python main.py --sanity   # quick sanity check before the formal run
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import plots
from dataset import build_subsets
from evaluate import compute_metrics, gnb_baseline, uncertainty_from_alpha
from model import EvidentialCNN
from train import edl_loss, train


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _flatten(subset):
    return np.stack([subset[i][0].numpy().reshape(-1) for i in range(len(subset))])


def _labels(subset):
    return np.array([subset[i][1] for i in range(len(subset))])


def sanity_check():
    set_seed(config.SEED)
    device = get_device()
    print(f"[sanity] device = {device}")

    train_sub, val_sub, test_sub = build_subsets()
    print(f"[sanity] subset sizes = train {len(train_sub)} / val {len(val_sub)} / test {len(test_sub)}")

    loader = DataLoader(train_sub, batch_size=config.BATCH_SIZE, shuffle=True)
    model = EvidentialCNN().to(device)

    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    alpha, logits = model(x)
    print(f"[sanity] input {tuple(x.shape)} / labels {tuple(y.shape)} / alpha {tuple(alpha.shape)}")
    assert alpha.shape == (x.shape[0], config.N_CLASSES)
    assert bool((alpha >= 1.0).all()), "alpha must be >= 1"

    loss = edl_loss(alpha, y, anneal=0.5)
    loss.backward()
    print(f"[sanity] one forward+backward ok, loss = {loss.item():.4f}")

    m = compute_metrics(y.cpu().numpy(),
                        alpha.argmax(-1).cpu().numpy(),
                        alpha.detach().cpu().numpy())
    print(f"[sanity] metrics computable, batch accuracy = {m['accuracy']:.4f}")

    tmp = os.path.join(config.RESULTS_DIR, "_sanity_model.pt")
    torch.save(model.state_dict(), tmp)
    model2 = EvidentialCNN()
    model2.load_state_dict(torch.load(tmp, map_location="cpu"))
    os.remove(tmp)
    print("[sanity] model save/reload ok")
    print("[sanity] PASSED")


def run():
    set_seed(config.SEED)
    device = get_device()
    print(f"[device] {device}")

    # 1. Data
    train_sub, val_sub, test_sub = build_subsets()
    n_train, n_val, n_test = len(train_sub), len(val_sub), len(test_sub)
    print(f"[data] train={n_train} val={n_val} test={n_test}")

    train_loader = DataLoader(train_sub, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_sub, batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. Model
    model = EvidentialCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params}")

    # 3. Train (fit)
    history, best_val_acc, fit_time = train(model, train_loader, val_loader, device)
    print(f"[train] done in {fit_time:.2f}s, best val acc = {best_val_acc:.4f}")

    # 4. Inference + metrics on the test set
    model.eval()
    all_alpha, all_pred, all_true = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            alpha, _ = model(x)
            all_alpha.append(alpha.cpu())
            all_pred.append(alpha.argmax(-1).cpu())
            all_true.append(y)
    infer_time = time.time() - t0

    alpha = torch.cat(all_alpha).numpy()
    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()

    metrics = compute_metrics(y_true, y_pred, alpha)
    metrics["fit_time_s"] = float(fit_time)
    metrics["inference_time_s"] = float(infer_time)
    metrics["n_train"] = n_train
    metrics["n_val"] = n_val
    metrics["n_test"] = n_test
    metrics["best_val_acc"] = float(best_val_acc)
    metrics["n_params"] = n_params
    metrics["device"] = str(device)

    # Gaussian Naive Bayes baseline (classical reference) on flattened pixels
    metrics["gnb_baseline"] = gnb_baseline(_flatten(train_sub), _labels(train_sub),
                                           _flatten(test_sub), _labels(test_sub))

    # 5. Save results
    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(config.RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    np.savez(os.path.join(config.RESULTS_DIR, "predictions.npz"),
             y_true=y_true, y_pred=y_pred, alpha=alpha)
    with open(os.path.join(config.RESULTS_DIR, "config.json"), "w") as f:
        json.dump({
            "seed": config.SEED, "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "batch_size": config.BATCH_SIZE, "epochs": config.EPOCHS,
            "learning_rate": config.LEARNING_RATE, "annealing_steps": config.ANNEALING_STEPS,
        }, f, indent=2)
    torch.save(model.state_dict(), os.path.join(config.RESULTS_DIR, "model.pt"))

    # 6. Figures
    plots.plot_confusion_matrix(np.asarray(metrics["confusion_matrix"]),
                                os.path.join(config.FIGURES_DIR, "confusion_matrix.png"))
    plots.plot_per_class_f1(metrics["per_class"],
                            os.path.join(config.FIGURES_DIR, "per_class_f1.png"))
    plots.plot_training_curves(history,
                               os.path.join(config.FIGURES_DIR, "training_curves.png"))
    unc = uncertainty_from_alpha(alpha)
    plots.plot_epistemic_uncertainty(unc["epistemic"], y_pred == y_true,
                                     os.path.join(config.FIGURES_DIR, "epistemic_uncertainty.png"))

    print("[done] results saved to results/, figures saved to figures/")
    print(f"[done] test accuracy = {metrics['accuracy']:.4f}, macro-F1 = {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true", help="run the sanity check only")
    args = parser.parse_args()
    sanity_check() if args.sanity else run()
