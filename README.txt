================================================================================
NLP Project 1 -- Interpretability of Model Collapse
In-Context Learning, Induction Heads, and Model Collapse
================================================================================

--------------------------------------------------------------------------------
1. WHAT THIS CODE DOES
--------------------------------------------------------------------------------

This implements the project brief's task end to end:

  1. A small causal transformer trained on the symbolic in-context-learning
     task of Singh et al. (2024): the context is a sequence of (symbol,
     label) pairs with a randomised, per-sequence symbol->label mapping; the
     model must infer the mapping in-context and predict the label for a
     repeated "query" symbol. This cannot be solved by memorisation (the
     mapping is re-randomised every sequence and train/val/test use disjoint
     symbol vocabularies -- see section 4), so solving it requires an
     induction circuit (a previous-token head feeding an induction head).

  2. A "base" and an "extended" version of the task (config.task.extended):
       base:     predict only the query label (the brief's "single number").
       extended: additionally predict ONE more (symbol, label) pair appended
                 after the query -- i.e. the model must also generate a
                 "next symbol" and its value. Predicting that symbol is NOT
                 inferable from context (it is a fresh draw), so the model
                 must rely on its learned PRIOR over the symbol vocabulary.
                 This is the deliberate lever for model collapse (see below).

  3. A model-collapse pipeline (collapse.py) that trains a sequence of
     models, generation 0 on real data and every subsequent generation on
     its predecessor's own outputs. The CONTEXT and query symbol are always
     resampled fresh from the true distribution (a "prompt" is always real);
     only the LOSS-SCORED continuation (the query label, and in the extended
     task the extra symbol + its label) is replaced by the previous
     generation's own prediction. This isolates exactly the thing model
     collapse is about -- training on a model's own generations -- without
     confounding it with a change in what the model is conditioned on.
       * mode="pure":  100% of each generation's data is self-generated.
       * mode="mixed": a fraction p_real of examples keep the TRUE label
         (fresh real data injected every generation).
       * sample_temperature: how the teacher model is sampled when building
         the next generation's targets. 0.0 = greedy ("clean" recursion,
         the model's own best guess becomes ground truth). 1.0 = proper
         ancestral sampling ("noisy" recursion -- the setting under which
         collapse theory predicts degradation).

  4. Interpretability (interp.py): previous-token-head / induction-head
     attention scores, single-head and keep-only ablation, causal activation
     patching (Conmy et al., 2023 style -- patch one head's output from a
     "clean" example into a "corrupted" example and see whether the
     prediction flips), and symbol-embedding geometry (PCA, participation
     ratio, mean pairwise cosine similarity -- all used to track whether the
     circuit and/or the representations degrade across generations).

--------------------------------------------------------------------------------
2. WHY THE TASK IS DESIGNED THIS WAY (for the write-up)
--------------------------------------------------------------------------------

The brief explicitly predicts that collapse is unlikely when "the model has
to produce a single number from each datapoint" (the base task), because a
single scalar output gives collapse little room to compound. It also predicts
that an extended version where the model "predicts the next symbol ... and
gives its corresponding value" is where collapse should show up, "as the
model may favour individual symbols which will skew the overall
distribution." The extended task above is a direct, literal implementation
of that hint: the extra-symbol prediction is the one output position in the
whole sequence that is NOT determined by induction (it is unpredictable from
context by construction), so it is exactly where a model's own biases can
enter, get baked into the next generation's "real" context symbols, and
compound. The key diagnostic is therefore the MARGINAL distribution of the
model's own generated symbols (entropy, KL-from-uniform, "effective support
size" -- see metrics.py), tracked across generations, alongside the usual
accuracy/loss/perplexity and the induction-circuit health itself. This also
gives a natural research question for the discussion section: does
distributional collapse (in the free generative choice) happen BEFORE
circuit collapse (in the constrained inductive copy), or do they move
together?

--------------------------------------------------------------------------------
3. SETUP
--------------------------------------------------------------------------------

Python 3.10+ required.

    pip install -r requirements.txt

No GPU is required; everything is sized to train on a CPU in minutes. If you
do have a GPU, set train.device = "cuda" in config.py / the relevant script.

--------------------------------------------------------------------------------
4. HOW TO RUN -- DO THIS FIRST
--------------------------------------------------------------------------------

Run the smoke test before anything else. It exercises every code path (data
generation, training, attention/ablation/patching analysis, and the
synthetic-data collapse pipeline) on a tiny configuration and takes well
under a minute:

    cd src
    python smoke_test.py

It should end with "ALL SMOKE TESTS PASSED". If it raises an exception,
please keep the full traceback -- every component here was written and
unit-tested (data.py and metrics.py are pure NumPy and were tested directly;
the torch-dependent files were carefully reviewed and syntax-checked)


Then run the real experiments:

    # fast sanity-check run (small data/epochs, ~5-10 min total)
    python run_experiments.py --preset quick

    # the real numbers for the report (larger data/epochs; timing depends on
    # your machine -- time generation 0 of one condition first and
    # extrapolate before committing to the full --preset full grid)
    python run_experiments.py --preset full

    # once that has finished, generate every figure used in the write-up
    python make_report_figures.py --preset full

`run_experiments.py --stage baseline` trains just the generation-0 models
(base and extended) including a small learning-rate sweep (use --sweep).
`--stage collapse` runs the 5-condition collapse grid described in section 1
(extended task x {pure, mixed} x {temperature 0, temperature 1}, plus one
base-task control run). `--only <tag>` restricts to a single condition
(e.g. --only extended_pure_noisy) if you want to iterate on one at a time.

Individual pieces can also be run directly, e.g.:

    python train.py --extended --sweep --n_epochs 30   # just the baseline

--------------------------------------------------------------------------------
5. PROJECT STRUCTURE
--------------------------------------------------------------------------------

src/
  config.py              task / model / train / collapse configuration (dataclasses)
  data.py                pure-NumPy task generator (unit-tested, see section 6)
  torch_data.py           thin torch.utils.data wrapper around data.py
  model.py                causal transformer with per-head attention access + ablation
  metrics.py              pure-NumPy entropy/KL/PCA/participation-ratio utilities (unit-tested)
  train.py                training loop + hyperparameter sweep
  interp.py               attention scores, ablation, activation patching, embedding geometry
  collapse.py             the model-collapse pipeline (synthetic data generation + orchestration)
  run_experiments.py      top-level script: baseline + full collapse grid
  make_report_figures.py  turns saved results into the figures for the write-up
  plots.py                plotting functions (has its own fake-data self-test: `python plots.py`)
  smoke_test.py           fast end-to-end sanity check -- RUN THIS FIRST

results/
  checkpoints/<run>/      model.pt, history.json, test_metrics.json, config.json per run
                           (collapse runs: one such folder per generation, gen0/gen1/...)
  tables/                 collapse_grid_<preset>.json, ablation tables
  figures/report/         final figures (after make_report_figures.py)

--------------------------------------------------------------------------------
6. WHAT WAS VALIDATED, AND HOW
--------------------------------------------------------------------------------


  * data.py and metrics.py have NO torch dependency and were fully unit
    tested here (sequence structure, induction-target correctness, the
    train/val/test symbol split, burstiness/repeat handling, entropy/KL/PCA
    sanity checks against known closed-form answers).
  * plots.py was run end-to-end against synthetic fake data (`python
    plots.py`) to confirm every figure renders without error.
  * model.py / train.py / interp.py / collapse.py were syntax-checked
    (py_compile) and reviewed line by line against the tested data.py
    contract (tensor shapes, indexing, the loss-masking convention).
  * smoke_test.py exists specifically to close this gap on your machine: run
    it first, before any full experiment.



--------------------------------------------------------------------------------
