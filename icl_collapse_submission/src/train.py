"""
Training loop for a single generation's model, plus a small hyperparameter
sweep utility. Can be used as a library (imported by collapse.py) or run
directly as a script for the baseline (generation-0) model.
"""

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ExperimentConfig, TaskConfig, ModelConfig, TrainConfig
from data import make_symbol_split
from torch_data import ICLDataset, masked_cross_entropy, masked_accuracy
from model import InductionTransformer, count_parameters


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at_step(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def evaluate(model, loader, device, query_label_idx: int = None) -> dict:
    """query_label_idx: if given, the index (into the (T-1)-length targets/logits
    axis) of the query-label position, as ALSO separately scored, in addition to
    the usual loss-mask-pooled composite metrics.

    This separate query_label_* breakdown matters because, for the EXTENDED
    task, the composite metrics pool the query_label position together with
    extra_symbol/extra_label -- which are BY DESIGN fresh, independently random
    draws used to probe the model's own output prior (see data.py's module
    docstring), not positions a model can ever learn to predict above roughly
    chance level. Pooling them with query_label (the one position actually
    solvable by in-context induction) caps the extended task's composite
    accuracy well below 100% even with a perfect induction circuit, and adds
    pure label noise to whatever metric is used to pick the "best" checkpoint /
    decide early stopping. query_label_idx lets callers use the clean,
    uncapped signal instead. For the base task (which only ever scores the
    query_label position) the two are numerically identical."""
    model.eval()
    tot_loss, tot_correct, tot_count, n_batches = 0.0, 0, 0, 0
    q_tot_loss, q_tot_correct, q_tot_count = 0.0, 0, 0
    with torch.no_grad():
        for input_ids, targets, loss_mask in loader:
            input_ids, targets, loss_mask = input_ids.to(device), targets.to(device), loss_mask.to(device)
            logits = model(input_ids)
            loss = masked_cross_entropy(logits, targets, loss_mask)
            preds = logits.argmax(dim=-1)
            correct = ((preds == targets) & loss_mask).sum().item()
            count = loss_mask.sum().item()
            tot_loss += loss.item() * count
            tot_correct += correct
            tot_count += count
            n_batches += 1

            if query_label_idx is not None:
                q_logits = logits[:, query_label_idx, :]
                q_targets = targets[:, query_label_idx]
                q_loss = F.cross_entropy(q_logits, q_targets, reduction="mean")
                q_correct = (q_logits.argmax(dim=-1) == q_targets).sum().item()
                q_tot_loss += q_loss.item() * q_targets.shape[0]
                q_tot_correct += q_correct
                q_tot_count += q_targets.shape[0]

    mean_loss = tot_loss / max(1, tot_count)
    out = dict(
        loss=mean_loss,
        accuracy=tot_correct / max(1, tot_count),
        perplexity=math.exp(min(mean_loss, 20)),  # clip to avoid overflow early in training
    )
    if query_label_idx is not None:
        q_mean_loss = q_tot_loss / max(1, q_tot_count)
        out.update(
            query_label_loss=q_mean_loss,
            query_label_accuracy=q_tot_correct / max(1, q_tot_count),
            query_label_perplexity=math.exp(min(q_mean_loss, 20)),
        )
    return out


def train_one_model(
    exp_cfg: ExperimentConfig,
    train_dataset: ICLDataset,
    val_dataset: ICLDataset,
    test_dataset: ICLDataset,
    out_dir: str = None,
    verbose: bool = True,
    init_state_dict: dict = None,
):
    """Train a single InductionTransformer. Returns (model, history, test_metrics)."""
    set_seed(exp_cfg.train.seed)
    device = torch.device(exp_cfg.train.device)

    task = exp_cfg.task
    model = InductionTransformer(exp_cfg.model, vocab_size=task.vocab_size).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    if verbose:
        print(f"[train] model params: {count_parameters(model):,}")

    train_loader = DataLoader(train_dataset, batch_size=exp_cfg.train.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=exp_cfg.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=exp_cfg.train.batch_size, shuffle=False)

    # Index (into the (T-1)-length targets/logits axis) of the query-label
    # position -- see evaluate()'s docstring for why this is tracked
    # separately from the loss-mask-pooled composite metrics. All three
    # datasets share the same task config, so the same index applies to each.
    query_label_idx = train_dataset.positions["query_label_pos"] - 1

    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_cfg.train.lr, weight_decay=exp_cfg.train.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * exp_cfg.train.n_epochs

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_ppl": [],
               "val_query_label_loss": [], "val_query_label_acc": [], "lr": []}
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    global_step = 0
    t0 = time.time()

    for epoch in range(exp_cfg.train.n_epochs):
        model.train()
        running_loss, running_correct, running_count = 0.0, 0, 0
        for input_ids, targets, loss_mask in train_loader:
            input_ids, targets, loss_mask = input_ids.to(device), targets.to(device), loss_mask.to(device)
            lr = lr_at_step(global_step, exp_cfg.train.lr, exp_cfg.train.warmup_steps, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            logits = model(input_ids)
            loss = masked_cross_entropy(logits, targets, loss_mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp_cfg.train.grad_clip)
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct = ((preds == targets) & loss_mask).sum().item()
                count = loss_mask.sum().item()
            running_loss += loss.item() * count
            running_correct += correct
            running_count += count
            global_step += 1

            if verbose and global_step % exp_cfg.train.log_every == 0:
                print(f"  step {global_step}/{total_steps}  lr={lr:.2e}  train_loss={loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device, query_label_idx=query_label_idx)
        history["epoch"].append(epoch)
        history["train_loss"].append(running_loss / max(1, running_count))
        history["train_acc"].append(running_correct / max(1, running_count))
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_ppl"].append(val_metrics["perplexity"])
        history["val_query_label_loss"].append(val_metrics["query_label_loss"])
        history["val_query_label_acc"].append(val_metrics["query_label_accuracy"])
        history["lr"].append(lr)

        if verbose:
            # val_qlabel_loss is printed explicitly because it (not the composite
            # val_loss) is what best-checkpoint/early-stopping below actually
            # selects on -- without it, which epoch gets restored as "best" can't
            # be read off the log, only guessed at from val_qlabel_acc (which
            # doesn't always peak at the same epoch as the loss, since loss also
            # rewards prediction confidence).
            print(
                f"[epoch {epoch+1}/{exp_cfg.train.n_epochs}] "
                f"train_loss={history['train_loss'][-1]:.4f} train_acc={history['train_acc'][-1]:.4f}  "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_ppl={val_metrics['perplexity']:.3f}  "
                f"val_qlabel_loss={val_metrics['query_label_loss']:.4f} "
                f"val_qlabel_acc={val_metrics['query_label_accuracy']:.4f}"
            )

        # Early stopping / best-checkpoint selection uses the QUERY-LABEL loss,
        # not the loss-mask-pooled composite loss. For the base task the two
        # are identical (only query_label is ever scored). For the extended
        # task the composite loss is diluted by extra_symbol/extra_label,
        # which are fundamentally unpredictable by construction (see
        # evaluate()'s docstring) -- using it as the stopping criterion adds
        # pure noise and can trigger early stopping (or pick a "best"
        # checkpoint) based on fluctuations that have nothing to do with
        # whether the induction circuit is actually improving.
        if val_metrics["query_label_loss"] < best_val_loss - 1e-5:
            best_val_loss = val_metrics["query_label_loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_no_improve = 0
            if verbose:
                print(f"  (new best checkpoint: epoch {best_epoch}, val_qlabel_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= exp_cfg.train.early_stop_patience:
                if verbose:
                    print(f"[train] early stopping at epoch {epoch+1} (no query_label val improvement for {epochs_no_improve} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch

    test_metrics = evaluate(model, test_loader, device, query_label_idx=query_label_idx)
    history["wall_clock_s"] = time.time() - t0
    history["n_params"] = count_parameters(model)

    if verbose:
        print(f"[train] restored checkpoint from epoch {best_epoch} (best val_qlabel_loss={best_val_loss:.4f})")
        print(f"[train] TEST: loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f} ppl={test_metrics['perplexity']:.3f}  "
              f"| query_label: loss={test_metrics['query_label_loss']:.4f} acc={test_metrics['query_label_accuracy']:.4f} "
              f"ppl={test_metrics['query_label_perplexity']:.3f}")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        exp_cfg.save(os.path.join(out_dir, "config.json"))

    return model, history, test_metrics


def build_datasets(exp_cfg: ExperimentConfig, split_seed: int = 0):
    train_syms, val_syms, test_syms = make_symbol_split(exp_cfg.task, seed=split_seed)
    train_ds = ICLDataset(exp_cfg.task, train_syms, exp_cfg.train.n_train, seed=exp_cfg.train.seed)
    val_ds = ICLDataset(exp_cfg.task, val_syms, exp_cfg.train.n_val, seed=exp_cfg.train.seed + 10_000)
    test_ds = ICLDataset(exp_cfg.task, test_syms, exp_cfg.train.n_test, seed=exp_cfg.train.seed + 20_000)
    return train_ds, val_ds, test_ds, (train_syms, val_syms, test_syms)


def hyperparameter_sweep(base_cfg: ExperimentConfig, grid: dict, out_dir: str, short_epochs: int = 10):
    """grid: e.g. {"lr": [1e-3, 3e-4, 1e-4], "d_model": [64, 128]}
    Runs every combination for a reduced number of epochs, ranks by val loss,
    and returns the best full ExperimentConfig (with the original n_epochs restored)."""
    import itertools

    os.makedirs(out_dir, exist_ok=True)
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    results = []

    for combo in combos:
        cfg = copy.deepcopy(base_cfg)
        tag_parts = []
        for k, v in zip(keys, combo):
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
            elif hasattr(cfg.train, k):
                setattr(cfg.train, k, v)
            else:
                raise ValueError(f"Unknown sweep key {k}")
            tag_parts.append(f"{k}={v}")
        tag = ",".join(tag_parts)
        cfg.train.n_epochs = short_epochs
        cfg.train.early_stop_patience = short_epochs  # no early stopping during the sweep

        print(f"\n[sweep] {tag}")
        train_ds, val_ds, test_ds, _ = build_datasets(cfg)
        _, history, test_metrics = train_one_model(cfg, train_ds, val_ds, test_ds, out_dir=None, verbose=False)
        results.append({"tag": tag, "combo": dict(zip(keys, combo)), "best_val_loss": min(history["val_loss"]), "test_acc": test_metrics["accuracy"]})
        print(f"[sweep] {tag} -> best_val_loss={results[-1]['best_val_loss']:.4f} test_acc={test_metrics['accuracy']:.4f}")

    results.sort(key=lambda r: r["best_val_loss"])
    with open(os.path.join(out_dir, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    best = results[0]
    best_cfg = copy.deepcopy(base_cfg)
    for k, v in best["combo"].items():
        if hasattr(best_cfg.model, k):
            setattr(best_cfg.model, k, v)
        else:
            setattr(best_cfg.train, k, v)
    print(f"\n[sweep] BEST: {best['tag']} (val_loss={best['best_val_loss']:.4f})")
    return best_cfg, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="../results/checkpoints/baseline")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--n_train", type=int, default=50_000)
    parser.add_argument("--sweep", action="store_true", help="run a small hyperparameter sweep first")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = ExperimentConfig(run_name="baseline_extended" if args.extended else "baseline")
    cfg.task.extended = args.extended
    cfg.train.n_epochs = args.n_epochs
    cfg.train.n_train = args.n_train
    cfg.train.seed = args.seed

    if args.sweep:
        grid = {"lr": [1e-3, 3e-4, 1e-4], "d_model": [64, 128]}
        cfg, sweep_results = hyperparameter_sweep(cfg, grid, out_dir=os.path.join(args.out_dir, "sweep"))
        cfg.train.n_epochs = args.n_epochs  # restore full training length for the final run

    train_ds, val_ds, test_ds, splits = build_datasets(cfg)
    model, history, test_metrics = train_one_model(cfg, train_ds, val_ds, test_ds, out_dir=args.out_dir)
    print("Done. Results saved to", args.out_dir)


if __name__ == "__main__":
    main()
