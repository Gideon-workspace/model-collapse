"""
Run this AFTER run_experiments.py has finished (baseline + collapse grid).
Loads the saved checkpoints / JSON results and produces every figure used in
the write-up, saved to results/figures/report/.

Usage:
    python make_report_figures.py --preset quick
    python make_report_figures.py --preset full
"""

import argparse
import json
import os

import numpy as np
import torch

from config import ExperimentConfig
from train import build_datasets
from model import InductionTransformer
from interp import collect_attention_stats, run_ablation_table, symbol_embedding_geometry
import plots

RESULTS_ROOT = "../results"


def load_model(ckpt_dir):
    cfg = ExperimentConfig.load(os.path.join(ckpt_dir, "config.json"))
    model = InductionTransformer(cfg.model, vocab_size=cfg.task.vocab_size)
    state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def make_baseline_figures(tag: str, preset: str, out_dir: str):
    ckpt_dir = os.path.join(RESULTS_ROOT, "checkpoints", f"baseline_{tag}_{preset}")
    if not os.path.exists(ckpt_dir):
        print(f"[skip] {ckpt_dir} not found")
        return

    with open(os.path.join(ckpt_dir, "history.json")) as f:
        history = json.load(f)
    plots.plot_training_curves(history, os.path.join(out_dir, f"baseline_{tag}_training_curves.png"),
                                f"Baseline ({tag}) training curves")

    model, cfg = load_model(ckpt_dir)
    device = torch.device("cpu")
    _, _, test_ds, _ = build_datasets(cfg)

    attn_stats = collect_attention_stats(model, test_ds, device)
    plots.plot_head_scores(attn_stats["previous_token_score"], attn_stats["induction_score"],
                            os.path.join(out_dir, f"baseline_{tag}_head_scores.png"),
                            f"Baseline ({tag}) attention head scores")

    rows = run_ablation_table(model, test_ds, device)
    with open(os.path.join(out_dir, f"baseline_{tag}_ablation.json"), "w") as f:
        json.dump(rows, f, indent=2)
    plots.plot_ablation(rows, os.path.join(out_dir, f"baseline_{tag}_ablation.png"),
                         f"Baseline ({tag}) head ablation")

    # Raw attention map for one example, strongest induction head
    best = max(((l, h) for l in range(cfg.model.n_layer) for h in range(cfg.model.n_head)),
               key=lambda lh: attn_stats["induction_score"][lh[0], lh[1]])
    with torch.no_grad():
        input_ids = test_ds.input_ids[:1]
        _, attns, _ = model(input_ids, return_attn=True)
    attn_matrix = attns[best[0]][0, best[1]].numpy()
    tokens = test_ds.tokens[0, :-1]
    n_sym = cfg.task.n_symbols
    tick_labels = [f"s{t}" if t < n_sym else f"l{t-n_sym}" for t in tokens]
    plots.plot_attention_heatmap(attn_matrix, os.path.join(out_dir, f"baseline_{tag}_attn_example_L{best[0]}H{best[1]}.png"),
                                  f"Baseline ({tag}) L{best[0]}H{best[1]} attention (example)", tick_labels=tick_labels)

    geom = symbol_embedding_geometry(model, cfg.task)
    print(f"[{tag}] primary induction head: L{best[0]}H{best[1]} (score={attn_stats['induction_score'][best]:.3f}), "
          f"embedding participation_ratio={geom['participation_ratio']:.2f}, "
          f"mean_cosine_sim={geom['mean_cosine_similarity']:.3f}")


def make_collapse_figures(preset: str, out_dir: str):
    grid_path = os.path.join(RESULTS_ROOT, "tables", f"collapse_grid_{preset}.json")
    if not os.path.exists(grid_path):
        print(f"[skip] {grid_path} not found")
        return
    with open(grid_path) as f:
        grid = json.load(f)

    extended_tags = {k: v for k, v in grid.items() if "extended" in k}
    if extended_tags:
        # NOTE: test_query_label_* (not the composite test_acc/test_loss) is
        # what should be plotted here. For the extended task, test_acc pools
        # query_label together with extra_symbol/extra_label, which are BY
        # DESIGN fundamentally unpredictable from context (fresh, independently
        # random draws used to probe the model's prior -- see data.py's module
        # docstring); including them caps composite accuracy well below 100%
        # even with a perfect induction circuit and has nothing to do with
        # whether the circuit itself is degrading under collapse. Falls back to
        # the old keys for any pre-existing results.json that predates this field.
        acc_key = "test_query_label_acc" if all("test_query_label_acc" in r for r in next(iter(extended_tags.values()))) else "test_acc"
        loss_key = "test_query_label_loss" if all("test_query_label_loss" in r for r in next(iter(extended_tags.values()))) else "test_loss"
        ppl_key = "test_query_label_ppl" if all("test_query_label_ppl" in r for r in next(iter(extended_tags.values()))) else "test_ppl"
        plots.plot_collapse_curves(extended_tags, acc_key, "test accuracy (query label)",
                                    os.path.join(out_dir, "collapse_test_acc.png"), "Test accuracy across generations")
        plots.plot_collapse_curves(extended_tags, loss_key, "test loss (query label)",
                                    os.path.join(out_dir, "collapse_test_loss.png"), "Test loss across generations")
        plots.plot_collapse_curves(extended_tags, ppl_key, "test perplexity (query label)",
                                    os.path.join(out_dir, "collapse_test_ppl.png"), "Test perplexity across generations")

        n_symbols = None
        any_tag = next(iter(extended_tags))
        try:
            cfg = ExperimentConfig.load(os.path.join(RESULTS_ROOT, "checkpoints", f"collapse_{any_tag}_{preset}", "gen0", "config.json"))
            n_symbols = cfg.task.n_symbols
        except Exception:
            pass
        uniform_bits = np.log2(n_symbols) if n_symbols else None

        plots.plot_collapse_curves(extended_tags, "extra_symbol_entropy_bits", "entropy of generated-symbol marginal (bits)",
                                    os.path.join(out_dir, "collapse_symbol_entropy.png"),
                                    "Distributional collapse: entropy of the model's own symbol prior",
                                    true_random_baseline=uniform_bits)
        plots.plot_collapse_curves(extended_tags, "extra_symbol_kl_from_uniform_bits", "KL(model || uniform) (bits)",
                                    os.path.join(out_dir, "collapse_symbol_kl.png"),
                                    "Distributional collapse: KL divergence from uniform")
        plots.plot_collapse_curves(extended_tags, "extra_symbol_invalid_frac", "P(model predicts a label token where a symbol was expected)",
                                    os.path.join(out_dir, "collapse_invalid_frac.png"), "Ontology-violation rate across generations")

        for tag, results in extended_tags.items():
            ind = np.array([r["induction_score"] for r in results])  # [n_gen, n_layer, n_head]
            best_lh = np.unravel_index(np.argmax(ind[0]), ind[0].shape)
            vals = ind[:, best_lh[0], best_lh[1]]
            extended_tags[tag] = [dict(r, primary_induction_score=float(v)) for r, v in zip(results, vals)]
        plots.plot_collapse_curves(extended_tags, "primary_induction_score", "induction score of strongest head",
                                    os.path.join(out_dir, "collapse_induction_score.png"),
                                    "Does the induction circuit itself degrade?")

        plots.plot_collapse_curves(extended_tags, "participation_ratio", "embedding participation ratio",
                                    os.path.join(out_dir, "collapse_embedding_participation_ratio.png"),
                                    "Symbol-embedding representation collapse")

    if "base_pure_noisy_control" in grid and "extended_pure_noisy" in grid:
        combined = {"base_pure_noisy_control": grid["base_pure_noisy_control"],
                    "extended_pure_noisy": grid["extended_pure_noisy"]}
        # Use the query_label-specific accuracy (falls back to the composite key
        # for older results) so base and extended are compared on the SAME
        # position -- base only ever scores query_label, so comparing it
        # against extended's composite (which also includes the
        # unpredictable-by-design extra_symbol/extra_label positions) would
        # understate the extended model unfairly.
        cmp_key = "test_query_label_acc" if all("test_query_label_acc" in r for r in combined["extended_pure_noisy"]) else "test_acc"
        plots.plot_collapse_curves(combined, cmp_key, "test accuracy (query label)",
                                    os.path.join(out_dir, "collapse_base_vs_extended.png"),
                                    "Base task vs extended task under pure recursive (noisy) collapse")

    # Embedding PCA before/after for all four extended conditions (not just
    # the noisy ones) -- pure_clean turns out to be the ONE condition that
    # does not collapse in embedding space (see collapse_embedding_participation_ratio.png),
    # so leaving it out was hiding the most informative contrast.
    for tag in ["extended_pure_clean", "extended_pure_noisy", "extended_mixed_clean", "extended_mixed_noisy"]:
        if tag not in grid:
            continue
        gens = sorted(int(r["generation"]) for r in grid[tag])
        gen0_dir = os.path.join(RESULTS_ROOT, "checkpoints", f"collapse_{tag}_{preset}", f"gen{gens[0]}")
        genN_dir = os.path.join(RESULTS_ROOT, "checkpoints", f"collapse_{tag}_{preset}", f"gen{gens[-1]}")
        if not (os.path.exists(gen0_dir) and os.path.exists(genN_dir)):
            continue
        model0, cfg0 = load_model(gen0_dir)
        modelN, cfgN = load_model(genN_dir)
        geom0 = symbol_embedding_geometry(model0, cfg0.task)
        geomN = symbol_embedding_geometry(modelN, cfgN.task)
        plots.plot_embedding_pca(geom0["X2d"], geomN["X2d"],
                                  os.path.join(out_dir, f"embedding_pca_{tag}.png"),
                                  f"{tag}: generation {gens[0]}", f"{tag}: generation {gens[-1]}")
        print(f"[{tag}] participation ratio gen{gens[0]}={geom0['participation_ratio']:.2f} "
              f"-> gen{gens[-1]}={geomN['participation_ratio']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    out_dir = os.path.join(RESULTS_ROOT, "figures", "report")
    os.makedirs(out_dir, exist_ok=True)

    for tag in ("base", "extended"):
        make_baseline_figures(tag, args.preset, out_dir)

    make_collapse_figures(args.preset, out_dir)

    print(f"\nAll figures written to {out_dir}/")


if __name__ == "__main__":
    main()
