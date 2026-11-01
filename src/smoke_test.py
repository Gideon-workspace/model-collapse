"""
Fast end-to-end smoke test (~20-30 seconds on CPU). Exercises every code path
(data generation, training, interpretability, activation patching, and the
model-collapse synthetic-data pipeline) on a tiny configuration, so that bugs
surface immediately rather than after a multi-minute full training run.

Run this FIRST, before any full experiment:
    python smoke_test.py

If anything raises an exception, copy the full traceback back for debugging.
"""

import os
import sys
import traceback

import numpy as np
import torch

from config import ExperimentConfig, TaskConfig, ModelConfig, TrainConfig, CollapseConfig
from data import make_symbol_split
from torch_data import ICLDataset, masked_cross_entropy, masked_accuracy
from model import InductionTransformer, count_parameters
from train import train_one_model, build_datasets
from interp import (
    collect_attention_stats, run_ablation_table, run_activation_patching,
    symbol_embedding_geometry, evaluate_dataset,
)
from collapse import (
    build_synthetic_dataset, measure_output_marginals, run_collapse_pipeline,
    train_and_summarize_gen0,
)


def tiny_config(extended: bool) -> ExperimentConfig:
    # val/test pool sizes must exceed n_ways by a margin (the extended task needs a
    # spare, never-in-context symbol per split) -- n_symbols=30 with these fractions
    # gives pools of 18 (train) / 6 (val) / 6 (test) against n_ways=3.
    # k_shot=2 + loss_on_context_labels=True (matching run_experiments.py's real
    # configs, not the old k_shot=1): exercises the burstiness/repeated-context-label
    # code path through the full torch pipeline, not just data.py's standalone
    # numpy unit tests.
    task = TaskConfig(n_symbols=30, n_labels=5, n_ways=3, k_shot=2, extended=extended,
                       val_frac=0.2, test_frac=0.2, loss_on_context_labels=True)
    model = ModelConfig(d_model=16, n_layer=2, n_head=2, d_mlp=32, max_seq_len=32)
    train = TrainConfig(n_train=256, n_val=64, n_test=64, batch_size=32, n_epochs=2,
                         lr=1e-3, warmup_steps=5, log_every=1000, device="cpu",
                         early_stop_patience=10)
    return ExperimentConfig(task=task, model=model, train=train, run_name="smoke")


def section(name):
    print(f"\n--- {name} ---")


def run():
    device = torch.device("cpu")

    section("1. Data generation (base task)")
    cfg = tiny_config(extended=False)
    # query_label (1) + one scored repeat per class now that k_shot=2 and
    # loss_on_context_labels=True (each class's 2nd-and-later occurrence in the
    # bursty context is also scored; k_shot=2 means exactly 1 repeat per class).
    expected_loss_positions = 1 + cfg.task.n_ways * (cfg.task.k_shot - 1)
    train_ds, val_ds, test_ds, splits = build_datasets(cfg)
    print("train/val/test sizes:", len(train_ds), len(val_ds), len(test_ds))
    print(f"input_ids shape: {train_ds.input_ids.shape} loss_mask sum per row (should be {expected_loss_positions}):",
          train_ds.loss_mask.sum(dim=1).unique())
    assert (train_ds.loss_mask.sum(dim=1) == expected_loss_positions).all()

    section("2. Model forward pass shapes")
    model = InductionTransformer(cfg.model, vocab_size=cfg.task.vocab_size).to(device)
    print("param count:", count_parameters(model))
    xb, yb, mb = train_ds[:8]
    logits, attns, head_outs = model(xb, return_attn=True)
    print("logits shape:", logits.shape, "expected:", (8, train_ds.input_ids.shape[1], cfg.task.vocab_size))
    assert logits.shape == (8, train_ds.input_ids.shape[1], cfg.task.vocab_size)
    assert len(attns) == cfg.model.n_layer
    print("attn[0] shape:", attns[0].shape, "expected (B,H,T,T) =",
          (8, cfg.model.n_head, train_ds.input_ids.shape[1], train_ds.input_ids.shape[1]))
    assert attns[0].shape == (8, cfg.model.n_head, train_ds.input_ids.shape[1], train_ds.input_ids.shape[1])
    # each attention row should sum to 1 (softmax over unmasked positions)
    row_sums = attns[0].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), row_sums

    loss = masked_cross_entropy(logits, yb, mb)
    print("initial loss (should be finite, roughly ln(vocab)=%.2f for an untrained model):" % np.log(cfg.task.vocab_size), loss.item())
    assert torch.isfinite(loss)

    section("3. Short training run (base task) -- loss should decrease")
    model, history, test_metrics = train_one_model(cfg, train_ds, val_ds, test_ds, out_dir=None, verbose=True)
    print("train_loss history:", [f"{x:.3f}" for x in history["train_loss"]])
    assert history["train_loss"][-1] <= history["train_loss"][0] + 1e-6, "loss did not decrease at all"
    print("test metrics:", test_metrics)

    section("4. Interpretability: attention stats")
    attn_stats = collect_attention_stats(model, test_ds, device)
    print("previous_token_score shape:", attn_stats["previous_token_score"].shape,
          "expected:", (cfg.model.n_layer, cfg.model.n_head))
    print("induction_score shape:", attn_stats["induction_score"].shape)
    assert attn_stats["previous_token_score"].shape == (cfg.model.n_layer, cfg.model.n_head)
    assert np.all(attn_stats["previous_token_score"] >= -1e-6) and np.all(attn_stats["previous_token_score"] <= 1 + 1e-6)
    assert np.all(attn_stats["induction_score"] >= -1e-6) and np.all(attn_stats["induction_score"] <= 1 + 1e-6)

    section("5. Interpretability: ablation table")
    rows = run_ablation_table(model, test_ds, device)
    print(f"ablation table has {len(rows)} rows (expected {1 + 2*cfg.model.n_layer*cfg.model.n_head})")
    assert len(rows) == 1 + 2 * cfg.model.n_layer * cfg.model.n_head
    baseline_row = [r for r in rows if r["type"] == "baseline"][0]
    print("baseline accuracy:", baseline_row["accuracy"])

    section("6. Interpretability: activation patching")
    patch_result = run_activation_patching(model, test_ds, device, layer=1, head=0, n_pairs=30, seed=0)
    print("activation patching result:", patch_result)
    assert 0.0 <= patch_result["patched_flip_to_clean_rate"] <= 1.0

    section("7. Embedding geometry")
    geom = symbol_embedding_geometry(model, cfg.task)
    print("participation_ratio:", geom["participation_ratio"], " mean_cosine_sim:", geom["mean_cosine_similarity"])
    print("X2d shape:", geom["X2d"].shape, "expected:", (cfg.task.n_symbols, 2))
    assert geom["X2d"].shape == (cfg.task.n_symbols, 2)

    section("8. Extended task + synthetic (model-collapse) data generation")
    cfg_ext = tiny_config(extended=True)
    # query_label + extra_symbol + extra_label (3) + one scored repeat per class.
    expected_loss_positions_ext = 3 + cfg_ext.task.n_ways * (cfg_ext.task.k_shot - 1)
    train_ds_e, val_ds_e, test_ds_e, splits_e = build_datasets(cfg_ext)
    model_e, history_e, test_metrics_e = train_one_model(cfg_ext, train_ds_e, val_ds_e, test_ds_e, out_dir=None, verbose=False)
    print("extended task test metrics:", test_metrics_e)
    assert (train_ds_e.loss_mask.sum(dim=1) == expected_loss_positions_ext).all()

    synth = build_synthetic_dataset(model_e, cfg_ext.task, splits_e[0], n_examples=100, device=device,
                                     temperature=1.0, p_real=0.3, seed=0, batch_size=50)
    print("synthetic dataset size:", len(synth), "input_ids shape:", synth.input_ids.shape)
    assert len(synth) == 100
    assert synth.loss_mask.sum(dim=1).unique().tolist() == [expected_loss_positions_ext]

    marg = measure_output_marginals(model_e, cfg_ext.task, splits_e[0], device, n_samples=200, temperature=1.0, seed=1)
    print("marginal stats keys:", sorted(marg.keys()))
    assert "extra_symbol_entropy_bits" in marg

    section("9. Full tiny collapse pipeline (2 generations, pure mode, retrain-from-scratch)")
    cfg_collapse = tiny_config(extended=True)
    cfg_collapse.collapse = CollapseConfig(n_generations=2, mode="pure", n_train_per_gen=128,
                                            run_ablation_every_gen=True, n_marginal_samples=100,
                                            retrain_from_scratch=True,
                                            continuation_n_epochs=1, continuation_early_stop_patience=1)
    cfg_collapse.train.n_epochs = 1
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        results = run_collapse_pipeline(cfg_collapse, out_dir=tmp, verbose=True)
    print(f"collapse pipeline produced {len(results)} generation summaries")
    assert len(results) == 2
    for r in results:
        assert "extra_symbol_entropy_bits" in r

    section("10. use_mlp=False (attention-only) architecture")
    cfg_noMLP = tiny_config(extended=False)
    cfg_noMLP.model.use_mlp = False
    model_noMLP = InductionTransformer(cfg_noMLP.model, vocab_size=cfg_noMLP.task.vocab_size).to(device)
    for block in model_noMLP.blocks:
        assert block.use_mlp is False
        assert not hasattr(block, "mlp"), "use_mlp=False should skip building the MLP sublayer entirely"
    train_ds_noMLP, val_ds_noMLP, test_ds_noMLP, _ = build_datasets(cfg_noMLP)
    xb, yb, mb = train_ds_noMLP[:8]
    logits_noMLP = model_noMLP(xb)
    assert logits_noMLP.shape == (8, train_ds_noMLP.input_ids.shape[1], cfg_noMLP.task.vocab_size)
    n_params_with_mlp = count_parameters(model)  # from section 2, use_mlp=True
    n_params_without_mlp = count_parameters(model_noMLP)
    print(f"param count with MLP: {n_params_with_mlp}, without MLP: {n_params_without_mlp}")
    assert n_params_without_mlp < n_params_with_mlp, "attention-only model should have strictly fewer params"

    section("11. shared_gen0 warm-start reuse across collapse conditions")
    cfg_shared = tiny_config(extended=True)
    cfg_shared.train.n_epochs = 1
    with tempfile.TemporaryDirectory() as tmp:
        gen0_dir = os.path.join(tmp, "shared_gen0")
        shared = train_and_summarize_gen0(cfg_shared, gen0_dir, verbose=False)
        for key in ("model", "history", "test_metrics", "gen_summary", "val_ds", "test_ds", "train_syms"):
            assert key in shared, f"train_and_summarize_gen0 result missing key {key!r}"

        # Two different collapse conditions (different mode/temperature) reusing
        # the SAME shared gen0 -- their gen0 test_acc/loss must come out identical
        # (it's the literal same trained model, just copied into each condition's
        # own gen0/ subdirectory), while each still runs its own generation >= 1.
        results_cond_a, results_cond_b = [], []
        for cond_name, mode, temp in (("cond_a", "pure", 1.0), ("cond_b", "mixed", 0.0)):
            cfg_cond = tiny_config(extended=True)
            cfg_cond.train.n_epochs = 1
            cfg_cond.collapse = CollapseConfig(
                n_generations=2, mode=mode, p_real=0.2, sample_temperature=temp,
                n_train_per_gen=128, run_ablation_every_gen=True, n_marginal_samples=100,
                retrain_from_scratch=False, continuation_n_epochs=1, continuation_early_stop_patience=1,
            )
            out_dir = os.path.join(tmp, cond_name)
            res = run_collapse_pipeline(cfg_cond, out_dir=out_dir, verbose=False, shared_gen0=shared)
            assert len(res) == 2
            (results_cond_a if cond_name == "cond_a" else results_cond_b).extend(res)

        gen0_a, gen0_b = results_cond_a[0], results_cond_b[0]
        assert gen0_a["test_acc"] == gen0_b["test_acc"] == shared["test_metrics"]["accuracy"], (
            "both conditions' gen0 should be the literal reused shared model"
        )
        print("shared gen0 test_acc reused identically across conditions:", gen0_a["test_acc"])

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        print("\nSMOKE TEST FAILED -- full traceback below:\n")
        traceback.print_exc()
        sys.exit(1)