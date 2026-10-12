# Evidential Deep Learning on MNIST

**CISC3024 Pattern Recognition — AI Assignment #2**

A recent, probability-based image classifier: **Evidential Deep Learning (EDL)**,
which places a Dirichlet distribution over class probabilities so that every
prediction carries an explicit measure of uncertainty. The entire workflow
(algorithm search, code, experiment, analysis, report) was produced by an AI
agent; no code was written by hand.

## Algorithm

- **Evidential Deep Learning** — Sensoy, M., Kaplan, L., & Kandemir, M. (2018).
  *Evidential Deep Learning to Quantify Classification Uncertainty.* NeurIPS 2018.
- The network outputs non-negative *evidence* `e_c = softplus(logit_c)` for each
  class, from which the Dirichlet concentration parameters are `α_c = e_c + 1`.
  The expected class probabilities are `p_c = α_c / Σ α_c`.
- Training minimises the type-II maximum-likelihood loss with a digamma-based
  term plus an annealed Kullback–Leibler regulariser towards a uniform Dirichlet.
- Because the output is a Dirichlet, uncertainty decomposes into **aleatoric**
  (data) and **epistemic** (model) components.

## Dataset

MNIST (28×28 grayscale digits), deterministic subsets with seed 42:

| Split | Size | Provenance |
|-------|------|------------|
| Train | 2,000 | official training split (shuffled) |
| Validation | 500 | official training split (shuffled) |
| Test | 1,000 | official test split (labels preserved, no leakage) |

## Environment setup

macOS with Apple Silicon. Requires Python **3.13** (PyTorch wheels are not
available for 3.14 in this environment).

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Execution

```bash
# 1. Sanity check (data, shapes, forward/backward, metrics, save/reload)
.venv/bin/python main.py --sanity

# 2. Run the full experiment (train + evaluate + save results/figures)
.venv/bin/python main.py

# 3. Regenerate the report (report.docx and report.pdf)
.venv/bin/python make_report.py
```

Results are written to `results/` (JSON metrics, history, predictions, model
checkpoint) and figures to `figures/`.

## Results

On the 1,000-image held-out test set (Apple M2, Metal/MPS backend):

| Metric | EDL (CNN) | Gaussian Naive Bayes (baseline) |
|--------|-----------|---------------------------------|
| Test accuracy | **91.00%** | 55.60% |
| Macro F1 | **0.9076** | 0.5259 |
| Fit time | 4.39 s | 0.004 s |
| Inference time (1,000 imgs) | 0.053 s | — |

Mean epistemic uncertainty: **0.301 nats** for misclassified images vs
**0.151 nats** for correct ones — the model is more evidential-uncertain exactly
when it is wrong.

## Report

- `report.docx` — Microsoft Word report
- `report.pdf` — PDF report
- `figures/` — confusion matrix, per-class F1, training curves, uncertainty histogram

## Project structure

```
main.py          # experiment orchestration + sanity check
config.py        # hyperparameters & paths (seed, splits, lr, epochs, ...)
dataset.py       # deterministic MNIST subsets (no test leakage)
model.py         # small CNN with evidential (Dirichlet) output head
train.py         # EDL loss + training loop
evaluate.py      # metrics + uncertainty decomposition + GNB baseline
plots.py         # report figures
make_report.py   # builds report.docx and report.pdf from results/
requirements.txt
```
