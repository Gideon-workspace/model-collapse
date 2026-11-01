"""
Synthetic in-context-learning task (symbol -> label induction), following the
few-shot classification setup of Chan et al. (2022) / Singh et al. (2024).

This module is pure NumPy (no torch dependency) so its logic can be unit
tested without a deep-learning framework installed. `torch_data.py` wraps
these functions in a torch.utils.data-compatible interface.

Sequence layout (token positions, 0-indexed):

    base:     c_0 y_0 c_1 y_1 ... c_{C-1} y_{C-1}  s_q  l_q
    extended: c_0 y_0 c_1 y_1 ... c_{C-1} y_{C-1}  s_q  l_q  s_e  l_e

where (c_i, y_i) are context (symbol, label) pairs, (s_q, l_q) is the query
pair (s_q repeats one of the context symbols; l_q is its associated label --
this is the position an induction head must solve), and (s_e, l_e), present
only in the "extended" task, is a BRAND NEW (never-before-seen-in-sequence)
symbol/label pair appended after the query. Predicting s_e exercises the
model's learned *prior* over the symbol marginal (it cannot be inferred from
context), which is the lever used for the model-collapse experiments: if a
model's prior over symbols is non-uniform, resampling synthetic training data
from it will skew the marginal further every generation.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import TaskConfig


def make_symbol_split(task: TaskConfig, seed: int = 0):
    """Disjoint train/val/test partition of the symbol vocabulary.

    Returns three 1-D int arrays of symbol ids. Validation and test sequences
    are built exclusively from symbols the model never saw during training,
    so good val/test accuracy can only come from genuine in-context
    induction, not from memorised per-symbol associations.
    """
    rng = np.random.default_rng(seed)
    all_syms = rng.permutation(task.n_symbols)
    n_test = int(round(task.n_symbols * task.test_frac))
    n_val = int(round(task.n_symbols * task.val_frac))
    if n_test < 1 or n_val < 1:
        raise ValueError("n_symbols too small for the requested val/test fractions")
    test_syms = np.sort(all_syms[:n_test])
    val_syms = np.sort(all_syms[n_test:n_test + n_val])
    train_syms = np.sort(all_syms[n_test + n_val:])
    assert len(set(train_syms) & set(val_syms)) == 0
    assert len(set(train_syms) & set(test_syms)) == 0
    assert len(set(val_syms) & set(test_syms)) == 0
    return train_syms, val_syms, test_syms


def _sample_distinct(rng: np.random.Generator, pool: np.ndarray, n: int, batch_size: int) -> np.ndarray:
    """Vectorised-ish sampling of `n` distinct values from `pool`, independently
    for each of `batch_size` rows. Returns int64 array [batch_size, n]."""
    if n > len(pool):
        raise ValueError(f"Requested {n} distinct symbols but pool only has {len(pool)}")
    out = np.empty((batch_size, n), dtype=np.int64)
    for b in range(batch_size):
        out[b] = rng.choice(pool, size=n, replace=False)
    return out


def generate_raw_batch(rng: np.random.Generator, task: TaskConfig, symbol_pool: np.ndarray, batch_size: int) -> dict:
    """Sample a batch of raw (symbol, label) structures from the TRUE generative
    process (i.e. everything exogenous / not coming from a trained model)."""
    C = task.n_ways
    Ctx = task.context_len_pairs
    B = batch_size
    pool = np.asarray(symbol_pool)

    classes = _sample_distinct(rng, pool, C, B)                     # [B, C] symbol ids
    labels_per_class = rng.integers(0, task.n_labels, size=(B, C))  # [B, C] label ids

    # Bursty context ordering: each class repeated k_shot times, then shuffled.
    reps = np.tile(np.arange(C), task.k_shot)  # length Ctx
    order = np.tile(reps, (B, 1)).copy()
    for b in range(B):
        rng.shuffle(order[b])

    ctx_symbols = np.take_along_axis(classes, order, axis=1)          # [B, Ctx]
    ctx_labels = np.take_along_axis(labels_per_class, order, axis=1)  # [B, Ctx]

    # Which context slots are a REPEAT of an earlier-in-context occurrence of
    # the same class (only possible when k_shot > 1).
    is_repeat = np.zeros((B, Ctx), dtype=bool)
    for b in range(B):
        seen = set()
        for i in range(Ctx):
            cls = order[b, i]
            if cls in seen:
                is_repeat[b, i] = True
            seen.add(cls)

    query_class_idx = rng.integers(0, C, size=B)
    query_symbol = classes[np.arange(B), query_class_idx]
    query_label = labels_per_class[np.arange(B), query_class_idx]

    # Position (within the Ctx context slots) of the LAST occurrence of the
    # query's class -- the position an induction head should copy the label from.
    match = (order == query_class_idx[:, None])
    idx_range = np.arange(Ctx)
    last_match_pos = np.argmax(match * (idx_range[None, :] + 1), axis=1)  # +1 avoids 0-index ambiguity
    assert np.all(match[np.arange(B), last_match_pos])

    out = dict(
        ctx_symbols=ctx_symbols,
        ctx_labels=ctx_labels,
        is_repeat=is_repeat,
        query_symbol=query_symbol,
        query_label=query_label,
        last_match_pos=last_match_pos,
    )

    if task.extended:
        extra_symbol = np.empty(B, dtype=np.int64)
        if len(pool) <= C:
            raise ValueError(
                f"Cannot sample a novel 'extra' symbol for the extended task: this "
                f"split's symbol pool has only {len(pool)} symbols, but n_ways={C} of "
                f"them are already used as context classes in every example, leaving "
                f"none spare. Increase n_symbols (or this split's val_frac/test_frac) "
                f"so every split's pool has at least n_ways + 1 symbols."
            )
        for b in range(B):
            avail = pool[~np.isin(pool, classes[b])]
            extra_symbol[b] = rng.choice(avail)
        extra_label = rng.integers(0, task.n_labels, size=B)
        out["extra_symbol"] = extra_symbol
        out["extra_label"] = extra_label

    return out


def assemble_tokens(task: TaskConfig, batch: dict) -> dict:
    """Turn a raw batch (from generate_raw_batch, or a synthetic variant built
    during the collapse pipeline) into model-ready token sequences.

    Returns:
        tokens:      int64 [B, T]   full sequence including the final target token
        input_ids:   int64 [B, T-1] tokens[:, :-1], what the model actually sees
        targets:     int64 [B, T-1] tokens[:, 1:], next-token targets
        loss_mask:   bool  [B, T-1] True at positions whose target is scored
        positions:   dict of useful token-index metadata (ints, not batched)
    """
    Ctx = task.context_len_pairs
    B = batch["ctx_symbols"].shape[0]
    T = task.seq_len_tokens

    tokens = np.zeros((B, T), dtype=np.int64)
    tokens[:, 0:2 * Ctx:2] = batch["ctx_symbols"]
    tokens[:, 1:2 * Ctx:2] = task.n_symbols + batch["ctx_labels"]
    tokens[:, 2 * Ctx] = batch["query_symbol"]
    tokens[:, 2 * Ctx + 1] = task.n_symbols + batch["query_label"]
    if task.extended:
        tokens[:, 2 * Ctx + 2] = batch["extra_symbol"]
        tokens[:, 2 * Ctx + 3] = task.n_symbols + batch["extra_label"]

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]
    loss_mask = np.zeros((B, T - 1), dtype=bool)

    # Query label is always scored.
    loss_mask[:, 2 * Ctx] = True

    if task.loss_on_context_labels:
        for i in range(Ctx):
            pos = 2 * i
            loss_mask[batch["is_repeat"][:, i], pos] = True

    if task.extended:
        loss_mask[:, 2 * Ctx + 1] = True  # predict extra_symbol from query_label
        loss_mask[:, 2 * Ctx + 2] = True  # predict extra_label from extra_symbol

    positions = dict(
        query_symbol_pos=2 * Ctx,
        query_label_pos=2 * Ctx + 1,
    )
    if task.extended:
        positions["extra_symbol_pos"] = 2 * Ctx + 2
        positions["extra_label_pos"] = 2 * Ctx + 3

    return dict(
        tokens=tokens,
        input_ids=input_ids,
        targets=targets,
        loss_mask=loss_mask,
        positions=positions,
        last_match_pos=batch["last_match_pos"],  # position within ctx (0..Ctx-1), not token index
    )


def generate_batch(rng: np.random.Generator, task: TaskConfig, symbol_pool: np.ndarray, batch_size: int) -> dict:
    """Convenience: raw batch + assembly in one call."""
    raw = generate_raw_batch(rng, task, symbol_pool, batch_size)
    return assemble_tokens(task, raw)
