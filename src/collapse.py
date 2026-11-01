"""
Model-collapse pipeline.

Methodology (mirrors Shumailov et al., 2023/2024, adapted to this task):

  * Generation 0 is trained on TRUE data (fresh, uniformly random
    context/query/extra symbols and labels -- see data.py).
  * For generation n >= 1, the CONTEXT and QUERY SYMBOL of every training
    example are still drawn fresh from the true exogenous distribution (the
    "prompt" is always real) -- only the LOSS-SCORED continuation (the query
    label, and, in the extended task, the extra symbol + its label) is
    replaced by generation (n-1)'s own model output, sampled with a
    configurable temperature. This isolates the thing model collapse is
    actually about: training on a model's own generations rather than on
    real data, without confounding it with a change in the conditioning
    distribution.
  * "pure" mode: 100% of each generation's training examples use the
    previous model's own output.
  * "mixed" mode: a fraction `p_real` of examples (chosen independently per
    example) use the TRUE label instead, representing continual injection of
    fresh real data alongside synthetic data.
  * `sample_temperature` controls how the teacher model is sampled when
    generating training targets for the next generation: temperature -> 0 is
    greedy/"clean" recursion (the previous model's own best guess becomes
    ground truth); temperature = 1 is proper ancestral sampling ("noisy"
    recursion, the setting under which collapse theory predicts degradation).

A FIXED true validation/test set (built once, from held-out symbols) is used
to evaluate every generation, so accuracy/loss/perplexity are comparable
across the whole pipeline.
"""

import copy
import json
import os

import numpy as np
import torch

from config import ExperimentConfig
from data import make_symbol_split, generate_raw_batch, assemble_tokens
from torch_data import ICLDataset
from model import InductionTransformer
from train import train_one_model
from interp import collect_attention_stats, run_ablation_table, symbol_embedding_geometry
from metrics import distribution_summary


class PrebuiltDataset(torch.utils.data.Dataset):
    """Duck-types ICLDataset so train.py / interp.py can consume synthetic
    (model-collapse) training data identically to real data."""

    def __init__(self, input_ids, targets, loss_mask, tokens, positions, last_match_pos):
        self.input_ids = torch.from_numpy(input_ids).long()
        self.targets = torch.from_numpy(targets).long()
        self.loss_mask = torch.from_numpy(loss_mask).bool()
        self.tokens = tokens
        self.positions = positions
        self.last_match_pos = last_match_pos

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return self.input_ids[idx], self.targets[idx], self.loss_mask[idx]


@torch.no_grad()
def teacher_generate(teacher_model, task, input_ids_prefix, device, temperature):
    """Autoregressively generate the loss-scored continuation tokens from a
    teacher model, given a prefix that ends at the query-symbol position.
    Caller is responsible for setting torch's RNG seed beforehand if
    reproducibility across calls is required.
    """
    seq = input_ids_prefix.to(device)
    generated = {}
    tok, _ = teacher_model.generate_next(seq, temperature=temperature)
    generated["query_label"] = tok
    seq = torch.cat([seq, tok.unsqueeze(1)], dim=1)
    if task.extended:
        tok2, _ = teacher_model.generate_next(seq, temperature=temperature)
        generated["extra_symbol"] = tok2
        seq = torch.cat([seq, tok2.unsqueeze(1)], dim=1)
        tok3, _ = teacher_model.generate_next(seq, temperature=temperature)
        generated["extra_label"] = tok3
        seq = torch.cat([seq, tok3.unsqueeze(1)], dim=1)
    return generated, seq


def build_synthetic_dataset(teacher_model, task, symbol_pool, n_examples, device,
                             temperature, p_real, seed, batch_size=512):
    """Build a PrebuiltDataset whose loss-scored targets come from `teacher_model`'s
    own outputs (with probability 1 - p_real per example) or the true generative
    process (with probability p_real per example)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    teacher_model.eval()

    qpos = task.context_len_pairs * 2  # query_symbol_pos

    all_tokens, all_loss_mask, all_last_match = [], [], []
    n_done = 0
    while n_done < n_examples:
        bs = min(batch_size, n_examples - n_done)
        raw = generate_raw_batch(rng, task, symbol_pool, bs)
        asm_true = assemble_tokens(task, raw)
        prefix = torch.from_numpy(asm_true["tokens"][:, : qpos + 1]).long()

        generated, _ = teacher_generate(teacher_model, task, prefix, device, temperature)

        use_real = rng.random(bs) < p_real
        final_tokens = asm_true["tokens"].copy()

        q_gen = generated["query_label"].cpu().numpy()
        final_tokens[~use_real, qpos + 1] = q_gen[~use_real]
        if task.extended:
            e_sym_gen = generated["extra_symbol"].cpu().numpy()
            e_lab_gen = generated["extra_label"].cpu().numpy()
            final_tokens[~use_real, qpos + 2] = e_sym_gen[~use_real]
            final_tokens[~use_real, qpos + 3] = e_lab_gen[~use_real]

        all_tokens.append(final_tokens)
        all_loss_mask.append(asm_true["loss_mask"])
        all_last_match.append(raw["last_match_pos"])
        n_done += bs

    tokens = np.concatenate(all_tokens, axis=0)
    loss_mask = np.concatenate(all_loss_mask, axis=0)
    last_match_pos = np.concatenate(all_last_match, axis=0)
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]

    positions = dict(query_symbol_pos=qpos, query_label_pos=qpos + 1)
    if task.extended:
        positions["extra_symbol_pos"] = qpos + 2
        positions["extra_label_pos"] = qpos + 3

    return PrebuiltDataset(input_ids, targets, loss_mask, tokens, positions, last_match_pos)


def _marginal_stats_for(raw_ids: np.ndarray, valid_lo: int, valid_hi: int, prefix: str) -> dict:
    """raw_ids: generated token ids that SHOULD lie in [valid_lo, valid_hi).
    Reports the categorical-distribution summary over the valid range plus the
    fraction of samples that fell outside it (a genuine model error / a
    second, independent symptom of degenerate generation under collapse)."""
    in_range = (raw_ids >= valid_lo) & (raw_ids < valid_hi)
    invalid_frac = 1.0 - in_range.mean() if len(raw_ids) > 0 else 0.0
    valid = raw_ids[in_range] - valid_lo
    if len(valid) == 0:
        summary = dict(entropy_bits=0.0, kl_from_uniform_bits=0.0, effective_support=0.0,
                        distinct_count=0, n_categories=valid_hi - valid_lo, n_samples=0)
    else:
        summary = distribution_summary(valid, valid_hi - valid_lo)
    summary["invalid_frac"] = float(invalid_frac)
    return {f"{prefix}_{k}": v for k, v in summary.items()}


@torch.no_grad()
def measure_output_marginals(model, task, symbol_pool, device, n_samples, temperature, seed, batch_size=512):
    """Sample many fresh (true) contexts, let the model generate its own
    continuation, and summarise the resulting marginal distributions. This is
    the key diagnostic for distributional collapse: under a healthy model the
    extra-symbol marginal (extended task only) should stay close to uniform
    over the symbol pool, since it is, by construction, unpredictable from
    context and should reflect the model's calibrated prior."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.eval()
    qpos = task.context_len_pairs * 2

    ql_all, es_all, el_all = [], [], []
    n_done = 0
    while n_done < n_samples:
        bs = min(batch_size, n_samples - n_done)
        raw = generate_raw_batch(rng, task, symbol_pool, bs)
        asm = assemble_tokens(task, raw)
        prefix = torch.from_numpy(asm["tokens"][:, : qpos + 1]).long()
        generated, _ = teacher_generate(model, task, prefix, device, temperature)
        ql_all.append(generated["query_label"].cpu().numpy())
        if task.extended:
            es_all.append(generated["extra_symbol"].cpu().numpy())
            el_all.append(generated["extra_label"].cpu().numpy())
        n_done += bs

    out = _marginal_stats_for(np.concatenate(ql_all), task.n_symbols, task.n_symbols + task.n_labels, "query_label")
    if task.extended:
        out.update(_marginal_stats_for(np.concatenate(es_all), 0, task.n_symbols, "extra_symbol"))
        out.update(_marginal_stats_for(np.concatenate(el_all), task.n_symbols, task.n_symbols + task.n_labels, "extra_label"))
    return out


def _summarize_generation(model, task, test_ds, train_syms, device, exp_cfg, generation: int,
                           n_train: int, history: dict, test_metrics: dict, gen_dir: str, verbose: bool):
    """Compute (and save) the interpretability + distributional-collapse diagnostics
    for one already-trained generation's model. Shared by generation 0 and every
    subsequent generation so the two code paths can't silently drift apart."""
    attn_stats = collect_attention_stats(model, test_ds, device)
    emb_geom = symbol_embedding_geometry(model, task)
    marginal_stats = measure_output_marginals(
        model, task, train_syms, device,
        n_samples=exp_cfg.collapse.n_marginal_samples, temperature=1.0, seed=777 + generation,
    )
    if exp_cfg.collapse.run_ablation_every_gen:
        ablation_rows = run_ablation_table(model, test_ds, device)
        with open(os.path.join(gen_dir, "ablation.json"), "w") as f:
            json.dump(ablation_rows, f, indent=2)

    gen_summary = dict(
        generation=generation,
        n_train=n_train,
        test_loss=test_metrics["loss"],
        test_acc=test_metrics["accuracy"],
        test_ppl=test_metrics["perplexity"],
        # query_label-specific breakdown (see train.evaluate()'s docstring):
        # for the extended task, test_acc/test_loss above are diluted by the
        # fundamentally-unpredictable extra_symbol/extra_label positions, so
        # THESE are the numbers that actually reflect whether the induction
        # circuit formed. For the base task they're identical to test_acc/test_loss.
        test_query_label_loss=test_metrics.get("query_label_loss", test_metrics["loss"]),
        test_query_label_acc=test_metrics.get("query_label_accuracy", test_metrics["accuracy"]),
        test_query_label_ppl=test_metrics.get("query_label_perplexity", test_metrics["perplexity"]),
        final_train_acc=history["train_acc"][-1],
        final_val_acc=history["val_acc"][-1],
        final_val_query_label_acc=history.get("val_query_label_acc", history["val_acc"])[-1],
        previous_token_score=np.asarray(attn_stats["previous_token_score"]).tolist(),
        induction_score=np.asarray(attn_stats["induction_score"]).tolist(),
        participation_ratio=emb_geom["participation_ratio"],
        mean_cosine_similarity=emb_geom["mean_cosine_similarity"],
        **marginal_stats,
    )
    with open(os.path.join(gen_dir, "gen_summary.json"), "w") as f:
        json.dump(gen_summary, f, indent=2, default=float)

    if verbose:
        print(f"[gen {generation}] test_acc(composite)={test_metrics['accuracy']:.4f} "
              f"test_acc(query_label)={gen_summary['test_query_label_acc']:.4f} test_loss={test_metrics['loss']:.4f}")
        if task.extended:
            print(f"[gen {generation}] extra_symbol entropy={gen_summary['extra_symbol_entropy_bits']:.3f} bits "
                  f"(uniform max={np.log2(task.n_symbols):.3f}), "
                  f"KL-from-uniform={gen_summary['extra_symbol_kl_from_uniform_bits']:.3f} bits, "
                  f"invalid_frac={gen_summary['extra_symbol_invalid_frac']:.4f}")
    return gen_summary


def train_and_summarize_gen0(exp_cfg: ExperimentConfig, out_dir: str, verbose: bool = True):
    """Train a generation-0 model to convergence (using exp_cfg.train AS GIVEN --
    pass in a generous n_epochs/early_stop_patience here) and compute its full
    gen_summary. Standalone so a single gen-0 model/result can be computed ONCE and
    reused (via `shared_gen0`) across several collapse conditions that start from
    the identical trained checkpoint (same task/model/train config and seed) --
    e.g. different collapse MODES applied on top of the same converged starting
    point, rather than wastefully retraining byte-identical copies of it.
    """
    os.makedirs(out_dir, exist_ok=True)
    task = exp_cfg.task
    device = torch.device(exp_cfg.train.device)
    train_syms, val_syms, test_syms = make_symbol_split(task, seed=0)
    val_ds = ICLDataset(task, val_syms, exp_cfg.train.n_val, seed=99_000)
    test_ds = ICLDataset(task, test_syms, exp_cfg.train.n_test, seed=99_001)
    train_ds = ICLDataset(task, train_syms, exp_cfg.collapse.n_train_per_gen, seed=exp_cfg.train.seed)

    model, history, test_metrics = train_one_model(exp_cfg, train_ds, val_ds, test_ds, out_dir=out_dir, verbose=verbose)
    gen_summary = _summarize_generation(model, task, test_ds, train_syms, device, exp_cfg, 0,
                                         len(train_ds), history, test_metrics, out_dir, verbose)
    return dict(model=model, history=history, test_metrics=test_metrics, gen_summary=gen_summary,
                val_ds=val_ds, test_ds=test_ds, train_syms=train_syms)


def run_collapse_pipeline(exp_cfg: ExperimentConfig, out_dir: str, verbose: bool = True, shared_gen0: dict = None):
    """shared_gen0: optional dict from train_and_summarize_gen0(), reused as this
    pipeline's generation 0 instead of training a fresh one. Its task/model/train
    config and seed MUST match exp_cfg's (the caller is responsible for that)."""
    os.makedirs(out_dir, exist_ok=True)
    task = exp_cfg.task
    device = torch.device(exp_cfg.train.device)

    if shared_gen0 is not None:
        train_syms = shared_gen0["train_syms"]
        val_ds, test_ds = shared_gen0["val_ds"], shared_gen0["test_ds"]
    else:
        train_syms, val_syms, test_syms = make_symbol_split(task, seed=0)
        val_ds = ICLDataset(task, val_syms, exp_cfg.train.n_val, seed=99_000)
        test_ds = ICLDataset(task, test_syms, exp_cfg.train.n_test, seed=99_001)

    gen_results = []
    prev_model = None

    for gen in range(exp_cfg.collapse.n_generations):
        if verbose:
            print(f"\n{'='*60}\nGeneration {gen} / {exp_cfg.collapse.n_generations - 1}  (mode={exp_cfg.collapse.mode})\n{'='*60}")

        gen_dir = os.path.join(out_dir, f"gen{gen}")
        os.makedirs(gen_dir, exist_ok=True)

        if gen == 0 and shared_gen0 is not None:
            if verbose:
                print("(reusing shared, pre-trained generation-0 model)")
            model = shared_gen0["model"]
            torch.save(model.state_dict(), os.path.join(gen_dir, "model.pt"))
            with open(os.path.join(gen_dir, "history.json"), "w") as f:
                json.dump(shared_gen0["history"], f, indent=2)
            with open(os.path.join(gen_dir, "test_metrics.json"), "w") as f:
                json.dump(shared_gen0["test_metrics"], f, indent=2)
            exp_cfg.save(os.path.join(gen_dir, "config.json"))
            gen_summary = dict(shared_gen0["gen_summary"])
            with open(os.path.join(gen_dir, "gen_summary.json"), "w") as f:
                json.dump(gen_summary, f, indent=2, default=float)
            if verbose:
                print(f"[gen 0] test_acc(composite)={shared_gen0['test_metrics']['accuracy']:.4f} "
                      f"test_acc(query_label)={gen_summary['test_query_label_acc']:.4f} "
                      f"test_loss={shared_gen0['test_metrics']['loss']:.4f}")
        else:
            cfg_gen = copy.deepcopy(exp_cfg)
            cfg_gen.train.seed = exp_cfg.train.seed + gen

            if gen == 0:
                train_ds = ICLDataset(task, train_syms, exp_cfg.collapse.n_train_per_gen, seed=cfg_gen.train.seed)
                init_state = None
            else:
                # Generations >= 1 use the smaller "continuation" budget and warm-start
                # from the previous generation by default (see CollapseConfig).
                cfg_gen.train.n_epochs = exp_cfg.collapse.continuation_n_epochs
                cfg_gen.train.early_stop_patience = exp_cfg.collapse.continuation_early_stop_patience
                p_real = exp_cfg.collapse.p_real if exp_cfg.collapse.mode == "mixed" else 0.0
                train_ds = build_synthetic_dataset(
                    prev_model, task, train_syms, exp_cfg.collapse.n_train_per_gen, device,
                    temperature=exp_cfg.collapse.sample_temperature, p_real=p_real, seed=cfg_gen.train.seed,
                )
                init_state = prev_model.state_dict() if not exp_cfg.collapse.retrain_from_scratch else None

            model, history, test_metrics = train_one_model(
                cfg_gen, train_ds, val_ds, test_ds, out_dir=gen_dir, verbose=verbose, init_state_dict=init_state,
            )
            gen_summary = _summarize_generation(model, task, test_ds, train_syms, device, exp_cfg, gen,
                                                 len(train_ds), history, test_metrics, gen_dir, verbose)

        gen_results.append(gen_summary)
        prev_model = model

    with open(os.path.join(out_dir, "collapse_results.json"), "w") as f:
        json.dump(gen_results, f, indent=2, default=float)

    return gen_results
