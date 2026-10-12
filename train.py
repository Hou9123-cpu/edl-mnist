"""Training loop using the Evidential Deep Learning loss.

The objective is the type-II maximum-likelihood loss of Sensoy et al. (2018),
optionally regularized by the Kullback-Leibler divergence between the predicted
Dirichlet and a uniform Dirichlet, with the KL weight annealed from 0 to 1.
"""
import time

import torch
import torch.nn.functional as F

from config import EPOCHS, LEARNING_RATE, ANNEALING_STEPS


def dirichlet_kl(alpha):
    """KL divergence KL[Dir(alpha) || Dir(1)] against a uniform Dirichlet."""
    K = alpha.shape[-1]
    S = alpha.sum(dim=-1)
    log_gamma_K = torch.lgamma(torch.tensor(float(K), dtype=alpha.dtype, device=alpha.device))
    kl = (
        torch.lgamma(S)
        - torch.lgamma(alpha).sum(dim=-1)
        - log_gamma_K
        + ((alpha - 1.0)
           * (torch.special.digamma(alpha) - torch.special.digamma(S).unsqueeze(-1))).sum(dim=-1)
    )
    return kl


def edl_loss(alpha, targets, anneal):
    """Evidential Deep Learning loss for a batch.

    Args:
        alpha: Dirichlet concentration parameters, shape (N, K).
        targets: integer class labels, shape (N,).
        anneal: scalar weight on the KL regularizer (0 to 1).
    """
    one_hot = F.one_hot(targets, num_classes=alpha.shape[-1]).float()
    S = alpha.sum(dim=-1, keepdim=True)
    # Type-II maximum-likelihood term: sum_c y_c (psi(S) - psi(alpha_c)).
    ml = (one_hot * (torch.special.digamma(S) - torch.special.digamma(alpha))).sum(dim=-1)
    # Remove the correct class's evidence before applying the regularizer.
    alpha_tilde = one_hot + (1.0 - one_hot) * alpha
    kl = dirichlet_kl(alpha_tilde)
    return ml.mean() + anneal * kl.mean()


def train(model, train_loader, val_loader, device):
    """Train ``model`` and return (history, best_val_acc, fit_time)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = -1.0, None

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        anneal = min(1.0, epoch / ANNEALING_STEPS)

        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            alpha, _ = model(x)
            loss = edl_loss(alpha, y, anneal)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = total / n
        val_loss, val_acc = _evaluate(model, val_loader, device, anneal)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    fit_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)  # restore best validation checkpoint
    return history, best_val_acc, fit_time


def _evaluate(model, loader, device, anneal):
    model.eval()
    total, n, correct = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            alpha, _ = model(x)
            total += edl_loss(alpha, y, anneal).item() * x.size(0)
            correct += (alpha.argmax(dim=-1) == y).sum().item()
            n += x.size(0)
    return total / n, correct / n
