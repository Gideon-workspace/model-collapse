"""
Top-level orchestration script. Runs:
  1. The baseline (generation-0-only) models for the base and extended tasks,
     including a small hyperparameter sweep.
  2. The model-collapse pipeline over a grid of conditions:
       extended task x {pure, mixed} x {temperature=0 ("clean"), temperature=1 ("noisy")}
     plus one base-task control run (pure, temperature=1), since the brief
     predicts collapse is unlikely to be visible on the single-number task.

Use --quick for a fast (~few minutes total) smoke run with small data/epochs
to sanity check the whole pipeline end-to-end before committing to the full
(--full) run, which is the one whose numbers should go in the report.

Examples:
    python run_experiments.py --quick
    python run_experiments.py --full
"""

import argparse
import copy
import json
import os
import time

from config import ExperimentConfig, TaskConfig, ModelConfig, TrainConfig, CollapseConfig
from train import build_datasets, train_one_model, hyperparameter_sweep
from collapse import run_collapse_pipeline, train_and_summarize_gen0

RESULTS_ROOT = "../results"


def base_experiment_config(extended: bool, preset: str) -> ExperimentConfig:
    # n_ways=4 (not 8): keeps sequences shorter (cheaper attention) and matches the
    # scale at which induction heads are known to form reliably within a modest
    # compute budget (c.f. Singh et al., 2024; Chan et al., 2022).
    # use_mlp=False: attention-only transformer, as Olsson et al. (2022) and Singh
    # et al. (2024) use for induction-head studies specifically -- an MLP gives the
    # model a competing (partial-memorisation) pathway that can delay or suppress
    # induction-head formation, and removing it is also noticeably cheaper per step.
    #
    # n_symbols=1000, k_shot=3, loss_on_context_labels=True (NOT the original
    # n_symbols=100, k_shot=1): a 200-epoch full run with the old values showed
    # train_acc shoot to >99% while val_qlabel_acc stayed pinned at ~30% and
    # val_loss/ppl actually BLEW UP (not just plateaued) as training continued --
    # the classic signature of in-weights memorisation outcompeting in-context
    # induction, not a slow pre-transition plateau. Chan et al. (2022) /
    # Singh et al. (2024) identify exactly the two levers that drive the
    # in-context-vs-in-weights balance: a LARGE symbol vocabulary (so no single
    # symbol's embedding gets enough repeated, idiosyncratic gradient pressure to
    # become a non-generalising shortcut -- with only 100 symbols over 50k
    # training examples each one was seen thousands of times) and BURSTINESS
    # (k_shot > 1, i.e. each class repeated within a single context), which the
    # old k_shot=1 setting removed entirely. loss_on_context_labels=True also
    # scores the repeated-class positions created by that burstiness, giving the
    # induction signal more genuinely-learnable supervision per step.
    task = TaskConfig(n_symbols=1000, n_labels=10, n_ways=4, k_shot=3, extended=extended,
                       loss_on_context_labels=True)
    model = ModelConfig(d_model=128, n_layer=2, n_head=4, d_mlp=512, max_seq_len=48, use_mlp=False)

    if preset == "quick":
        # Doubles as a FAST diagnostic for whether a task/config change actually
        # closes the train/val gap (not just a pipeline sanity check): small
        # n_train keeps each epoch sub-second, but n_epochs/patience are now high
        # enough to actually reveal the memorisation-vs-induction pattern (which
        # previously only showed up after ~25-30 epochs on the full preset).
        train = TrainConfig(n_train=3000, n_val=500, n_test=500, batch_size=64,
                             n_epochs=80, lr=1e-3, warmup_steps=50, early_stop_patience=30)
    else:
        # early_stop_patience=10 (NOT a generous value): a 200-epoch/patience=40
        # run's training curves showed this task does NOT have a long plateau
        # before a beneficial transition -- it has a short (~12 epoch) phase
        # where train and val genuinely improve TOGETHER, followed by a SHARP
        # overfitting cliff where train shoots to ~100% while val accuracy
        # actively drops and val loss explodes (classic catastrophic overfitting,
        # not a delayed transition). best_state is already restored by the
        # lowest val_qlabel_loss seen, so the final checkpoint was always "the
        # best so far" -- but a long patience window lets training wander deep
        # into the noisy post-cliff region for many epochs, where a spurious,
        # noise-driven dip in val_qlabel_loss can masquerade as a new "best" and
        # get selected over the real, cleaner pre-cliff optimum. A short patience
        # keeps the search confined to the good region and stops wasting an hour
        # of compute on a post-cliff phase that was never helping. n_epochs=200
        # stays as a safety ceiling; with this patience training should stop
        # within the first ~20-30 epochs.
        train = TrainConfig(n_train=50_000, n_val=2_000, n_test=2_000, batch_size=128,
                             n_epochs=200, lr=1e-3, warmup_steps=300, early_stop_patience=10)

    return ExperimentConfig(task=task, model=model, train=train,
                             run_name=f"{'extended' if extended else 'base'}_{preset}")


def run_baseline(preset: str, do_sweep: bool):
    for extended in (False, True):
        tag = "extended" if extended else "base"
        print(f"\n{'#'*70}\n# BASELINE: {tag} task ({preset})\n{'#'*70}")
        cfg = base_experiment_config(extended, preset)
        out_dir = os.path.join(RESULTS_ROOT, "checkpoints", f"baseline_{tag}_{preset}")

        if do_sweep:
            grid = {"lr": [1e-3, 3e-4, 1e-4]}
            sweep_dir = os.path.join(out_dir, "sweep")
            cfg, _ = hyperparameter_sweep(cfg, grid, out_dir=sweep_dir, short_epochs=max(3, cfg.train.n_epochs // 3))

        train_ds, val_ds, test_ds, splits = build_datasets(cfg)
        t0 = time.time()
        model, history, test_metrics = train_one_model(cfg, train_ds, val_ds, test_ds, out_dir=out_dir)
        # test_acc is the composite, loss-mask-pooled accuracy; for the extended
        # task it's diluted by the fundamentally-unpredictable extra_symbol/
        # extra_label positions (see train.evaluate()'s docstring), so
        # query_label_accuracy -- the one position actually solvable by
        # induction -- is the number that tells us whether the circuit formed.
        # They're identical for the base task.
        print(f"[baseline:{tag}] done in {time.time()-t0:.1f}s. "
              f"test_acc(composite)={test_metrics['accuracy']:.4f} "
              f"test_acc(query_label)={test_metrics['query_label_accuracy']:.4f}")


def collapse_conditions(preset: str):
    """Yields (tag, ExperimentConfig) for every condition in the collapse grid.

    IMPORTANT: cfg.train.n_epochs / early_stop_patience are left exactly as
    base_experiment_config(preset) sets them -- that is GENERATION 0's budget,
    and it must be the same generous budget the baseline uses to actually form
    the induction circuit (early-stopping or truncating it would reproduce the
    original bug). Only generations >= 1 use the separate, smaller
    continuation_n_epochs/continuation_early_stop_patience budget below, since
    those warm-start from an already-converged checkpoint (retrain_from_scratch
    =False) rather than rediscovering the circuit from scratch every generation.
    """
    if preset == "quick":
        n_generations, n_train_per_gen = 3, 1500
        continuation_n_epochs, continuation_early_stop_patience = 4, 4
    else:
        n_generations, n_train_per_gen = 8, 20_000
        continuation_n_epochs, continuation_early_stop_patience = 20, 8

    grid = [
        dict(extended=True, mode="pure", temperature=0.0, tag="extended_pure_clean"),
        dict(extended=True, mode="pure", temperature=1.0, tag="extended_pure_noisy"),
        dict(extended=True, mode="mixed", temperature=1.0, tag="extended_mixed_noisy"),
        dict(extended=True, mode="mixed", temperature=0.0, tag="extended_mixed_clean"),
        dict(extended=False, mode="pure", temperature=1.0, tag="base_pure_noisy_control"),
    ]

    for g in grid:
        cfg = base_experiment_config(g["extended"], preset)
        cfg.collapse = CollapseConfig(
            n_generations=n_generations,
            mode=g["mode"],
            p_real=0.2,
            sample_temperature=g["temperature"],
            retrain_from_scratch=False,
            n_train_per_gen=n_train_per_gen,
            run_ablation_every_gen=True,
            n_marginal_samples=2000 if preset == "full" else 300,
            continuation_n_epochs=continuation_n_epochs,
            continuation_early_stop_patience=continuation_early_stop_patience,
        )
        cfg.run_name = g["tag"]
        yield g["tag"], cfg


def run_collapse_grid(preset: str, only: str = None):
    """The 4 extended-task conditions all start from the IDENTICAL gen-0 setup
    (same task/model/train config and seed -- they only differ in collapse
    mode/temperature, which only affects generation >= 1). Rather than retrain
    that gen-0 model 4 times over (it was previously confirmed byte-identical
    across conditions in the quick-preset run), we train it ONCE here via
    train_and_summarize_gen0() and hand it to run_collapse_pipeline() as
    shared_gen0 for each of those 4 conditions. The base-task control condition
    is NOT extended, so it trains its own standalone gen0 as before."""
    all_results = {}
    shared_gen0 = None
    shared_gen0_cfg = base_experiment_config(True, preset)

    for tag, cfg in collapse_conditions(preset):
        if only is not None and only != tag:
            continue
        print(f"\n{'#'*70}\n# COLLAPSE PIPELINE: {tag} ({preset})\n{'#'*70}")
        out_dir = os.path.join(RESULTS_ROOT, "checkpoints", f"collapse_{tag}_{preset}")
        t0 = time.time()

        if cfg.task.extended:
            if shared_gen0 is None:
                print(f"\n--- Training SHARED extended-task generation 0 "
                      f"(reused by every extended-task collapse condition) ---")
                gen0_dir = os.path.join(RESULTS_ROOT, "checkpoints", f"collapse_shared_gen0_extended_{preset}")
                shared_gen0 = train_and_summarize_gen0(shared_gen0_cfg, gen0_dir, verbose=True)
            results = run_collapse_pipeline(cfg, out_dir=out_dir, verbose=True, shared_gen0=shared_gen0)
        else:
            results = run_collapse_pipeline(cfg, out_dir=out_dir, verbose=True)

        print(f"[collapse:{tag}] done in {time.time()-t0:.1f}s")
        all_results[tag] = results

    tables_dir = os.path.join(RESULTS_ROOT, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, f"collapse_grid_{preset}.json")
    # merge with any existing file so --only runs don't clobber previous conditions
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined results to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--stage", choices=["baseline", "collapse", "all"], default="all")
    parser.add_argument("--sweep", action="store_true", help="run a small LR sweep before the baseline run")
    parser.add_argument("--only", default=None, help="run only this one collapse condition tag")
    args = parser.parse_args()

    if args.stage in ("baseline", "all"):
        run_baseline(args.preset, do_sweep=args.sweep)
    if args.stage in ("collapse", "all"):
        run_collapse_grid(args.preset, only=args.only)


if __name__ == "__main__":
    main()
