"""
Plotting utilities for the report. No torch dependency (operates on saved
JSON / numpy results), so it can run anywhere, including just to preview
figure layout before a real training run finishes.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "pure_clean": "#1b9e77",
    "pure_noisy": "#d95f02",
    "mixed_noisy": "#7570b3",
    "mixed_clean": "#e7298a",
    "control": "#666666",
}


def _color_for_tag(tag: str) -> str:
    for key, c in COLORS.items():
        if key in tag:
            return c
    return "#333333"


def plot_training_curves(history: dict, out_path: str, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    epochs = history["epoch"]
    axes[0].plot(epochs, history["train_loss"], label="train", color="#1b9e77")
    axes[0].plot(epochs, history["val_loss"], label="val", color="#d95f02")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(frameon=False)
    axes[1].plot(epochs, history["train_acc"], label="train", color="#1b9e77")
    axes[1].plot(epochs, history["val_acc"], label="val", color="#d95f02")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_attention_heatmap(attn_matrix: np.ndarray, out_path: str, title: str, tick_labels=None):
    """attn_matrix: [T, T] attention weights for one (layer, head) and one example."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(attn_matrix, cmap="viridis", vmin=0, vmax=max(attn_matrix.max(), 1e-6))
    ax.set_xlabel("key position"); ax.set_ylabel("query position")
    ax.set_title(title)
    if tick_labels is not None and len(tick_labels) <= 24:
        ax.set_xticks(range(len(tick_labels))); ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tick_labels))); ax.set_yticklabels(tick_labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_head_scores(previous_token_score: np.ndarray, induction_score: np.ndarray, out_path: str, title: str):
    """Both arrays: [n_layer, n_head]."""
    n_layer, n_head = previous_token_score.shape
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for name, arr, ax in [("previous-token score", previous_token_score, axes[0]),
                           ("induction score", induction_score, axes[1])]:
        im = ax.imshow(arr, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_head)); ax.set_xticklabels([f"H{h}" for h in range(n_head)])
        ax.set_yticks(range(n_layer)); ax.set_yticklabels([f"L{l}" for l in range(n_layer)])
        ax.set_title(name)
        for l in range(n_layer):
            for h in range(n_head):
                ax.text(h, l, f"{arr[l,h]:.2f}", ha="center", va="center",
                         color="white" if arr[l, h] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle(title)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(rows: list, out_path: str, title: str):
    single = [r for r in rows if r["type"] == "single_ablation"]
    keep = [r for r in rows if r["type"] == "keep_only"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, data, name in [(axes[0], single, "single-head ablation (Δ accuracy)"),
                            (axes[1], keep, "keep-only this head (accuracy)")]:
        labels = [f"L{r['layer']}H{r['head']}" for r in data]
        values = [r["delta_acc"] if name.startswith("single") else r["accuracy"] for r in data]
        colors = ["#d95f02" if v < 0 else "#1b9e77" for v in values] if name.startswith("single") else "#7570b3"
        ax.bar(labels, values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title(name)
        ax.axhline(0, color="black", linewidth=0.6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_collapse_curves(tag_to_results: dict, metric_key: str, ylabel: str, out_path: str, title: str, true_random_baseline: float = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag, results in tag_to_results.items():
        gens = [r["generation"] for r in results]
        vals = [r[metric_key] for r in results]
        ax.plot(gens, vals, marker="o", label=tag, color=_color_for_tag(tag))
    if true_random_baseline is not None:
        ax.axhline(true_random_baseline, color="gray", linestyle="--", linewidth=1, label="uniform / chance")
    ax.set_xlabel("generation"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_collapse_curves_with_error_bars(tag_to_seed_results: dict, metric_key: str, ylabel: str, out_path: str, title: str):
    """tag_to_seed_results: {tag: [ [gen_summaries for seed0], [gen_summaries for seed1], ... ]}"""
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag, seed_runs in tag_to_seed_results.items():
        n_gen = len(seed_runs[0])
        gens = list(range(n_gen))
        vals = np.array([[run[g][metric_key] for g in range(n_gen)] for run in seed_runs])  # [n_seeds, n_gen]
        mean, std = vals.mean(axis=0), vals.std(axis=0)
        color = _color_for_tag(tag)
        ax.plot(gens, mean, marker="o", label=tag, color=color)
        ax.fill_between(gens, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_xlabel("generation"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_pca(X2d_before: np.ndarray, X2d_after: np.ndarray, out_path: str, title_before: str, title_after: str):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    axes[0].scatter(X2d_before[:, 0], X2d_before[:, 1], s=14, alpha=0.7, color="#1b9e77")
    axes[0].set_title(title_before)
    axes[1].scatter(X2d_after[:, 0], X2d_after[:, 1], s=14, alpha=0.7, color="#d95f02")
    axes[1].set_title(title_after)
    for ax in axes:
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-test with synthetic (fake) data -- verifies every plotting function
# runs end-to-end without needing a real training run.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out_dir = "../results/figures/_plot_selftest"
    os.makedirs(out_dir, exist_ok=True)

    fake_history = dict(epoch=list(range(20)),
                         train_loss=list(np.exp(-np.linspace(0, 2, 20)) + 0.05 * rng.normal(size=20)),
                         val_loss=list(np.exp(-np.linspace(0, 1.8, 20)) + 0.05 * rng.normal(size=20)),
                         train_acc=list(1 - np.exp(-np.linspace(0, 2, 20))),
                         val_acc=list(1 - np.exp(-np.linspace(0, 1.8, 20))))
    plot_training_curves(fake_history, f"{out_dir}/training_curves.png", "fake training curves")

    T = 18
    attn = rng.dirichlet(np.ones(T) * 0.3, size=T)
    attn = np.tril(attn)
    attn = attn / attn.sum(axis=1, keepdims=True)
    plot_attention_heatmap(attn, f"{out_dir}/attn_heatmap.png", "fake attention map")

    pts = rng.random((2, 4))
    ind = rng.random((2, 4))
    plot_head_scores(pts, ind, f"{out_dir}/head_scores.png", "fake head scores")

    rows = [dict(type="baseline", layer=None, head=None, accuracy=0.97, loss=0.1, delta_acc=0.0)]
    for l in range(2):
        for h in range(4):
            rows.append(dict(type="single_ablation", layer=l, head=h, accuracy=0.95 + 0.01 * rng.normal(),
                              loss=0.1, delta_acc=-0.02 + 0.01 * rng.normal()))
            rows.append(dict(type="keep_only", layer=l, head=h, accuracy=max(0, min(1, rng.random())),
                              loss=0.1, delta_acc=0.0))
    plot_ablation(rows, f"{out_dir}/ablation.png", "fake ablation")

    fake_results = {tag: [dict(generation=g, test_acc=1.0 - 0.02 * g - 0.01 * rng.random(),
                                extra_symbol_entropy_bits=6.3 - 0.3 * g)
                           for g in range(6)]
                     for tag in ["extended_pure_clean", "extended_pure_noisy", "extended_mixed_noisy"]}
    plot_collapse_curves(fake_results, "test_acc", "test accuracy", f"{out_dir}/collapse_acc.png", "fake collapse accuracy")
    plot_collapse_curves(fake_results, "extra_symbol_entropy_bits", "entropy (bits)",
                          f"{out_dir}/collapse_entropy.png", "fake entropy collapse", true_random_baseline=np.log2(100))

    seed_runs = {"extended_pure_noisy": [fake_results["extended_pure_noisy"], fake_results["extended_pure_noisy"]]}
    plot_collapse_curves_with_error_bars(seed_runs, "test_acc", "test accuracy",
                                          f"{out_dir}/collapse_acc_errbars.png", "fake collapse w/ error bars")

    X_before = rng.normal(size=(80, 16))
    X_after = rng.normal(size=(80, 16)) * 0.3  # tighter spread, simulating collapse
    from metrics import pca_2d
    X2d_before, _, _ = pca_2d(X_before)
    X2d_after, _, _ = pca_2d(X_after)
    plot_embedding_pca(X2d_before, X2d_after, f"{out_dir}/embedding_pca.png", "gen 0", "gen 7")

    print(f"All plot self-tests ran without error. Figures written to {out_dir}/")
