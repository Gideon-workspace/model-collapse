"""Thin torch.utils.data wrapper around the pure-NumPy generator in data.py.

Datasets here are *materialised* (fixed-size, generated once up front) rather
than streamed, so that train/val/test sets are reproducible and training
epochs have a well-defined size, as the brief requires ("you must use a
train-validation-test data split").
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from config import TaskConfig
from data import generate_batch


class ICLDataset(Dataset):
    def __init__(self, task: TaskConfig, symbol_pool: np.ndarray, n_examples: int, seed: int):
        rng = np.random.default_rng(seed)
        batch = generate_batch(rng, task, symbol_pool, n_examples)
        self.input_ids = torch.from_numpy(batch["input_ids"]).long()
        self.targets = torch.from_numpy(batch["targets"]).long()
        self.loss_mask = torch.from_numpy(batch["loss_mask"]).bool()
        self.tokens = batch["tokens"]  # kept as numpy for inspection / interpretability
        self.positions = batch["positions"]
        self.last_match_pos = batch["last_match_pos"]

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return self.input_ids[idx], self.targets[idx], self.loss_mask[idx]


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """logits: [B,T,V], targets: [B,T], loss_mask: [B,T] bool. Mean over masked positions only."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    mask_flat = loss_mask.reshape(B * T)
    losses = torch.nn.functional.cross_entropy(logits_flat, targets_flat, reduction="none")
    losses = losses[mask_flat]
    return losses.mean()


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == targets) & loss_mask
    return correct.sum().item() / loss_mask.sum().item()
