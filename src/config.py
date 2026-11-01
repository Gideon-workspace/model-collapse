"""
Central configuration for the ICL / induction-head / model-collapse project.

Everything that controls the task, the model, training, and the collapse
pipeline lives here so experiments are reproducible from a single file.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class TaskConfig:
    """Defines the symbolic in-context-learning task (Singh et al., 2024 style)."""

    # Vocabulary sizes
    n_symbols: int = 100       # total number of distinct "class" symbols in the world
    n_labels: int = 10         # total number of distinct label values (e.g. digits 0-9)

    # Few-shot structure
    n_ways: int = 8            # number of distinct classes shown per sequence (context length)
    k_shot: int = 1            # number of times each class is repeated in the context (burstiness)
                                # k_shot=1 -> no repeats, induction only needed at the query.
                                # k_shot>1 -> some context labels are also inductively predictable.

    # Base vs extended task
    extended: bool = False     # if True, append one fresh (novel symbol, novel label) pair
                                # after the query pair, and train on predicting BOTH tokens.

    # Whether to also put next-token loss on the repeated context label positions
    # (only has an effect when k_shot > 1; first occurrence of each class is never
    # predictable and is always excluded from the loss).
    loss_on_context_labels: bool = False

    # Train/val/test are defined by a DISJOINT split of the symbol vocabulary, so that
    # generalisation requires genuine in-context induction rather than memorised
    # per-symbol associations. Labels are shared across splits (only 10 of them).
    val_frac: float = 0.1
    test_frac: float = 0.1

    # Special token ids are placed after symbols and labels.
    @property
    def vocab_size(self) -> int:
        return self.n_symbols + self.n_labels

    def symbol_id(self, local_idx):
        return local_idx  # symbols occupy [0, n_symbols)

    def label_id(self, local_idx):
        return self.n_symbols + local_idx  # labels occupy [n_symbols, n_symbols+n_labels)

    @property
    def context_len_pairs(self) -> int:
        return self.n_ways * self.k_shot

    @property
    def seq_len_tokens(self) -> int:
        """Number of INPUT tokens (teacher-forced), i.e. length of the token sequence
        actually fed to the model. Does not include the final predicted token."""
        base = 2 * self.context_len_pairs + 2  # context pairs + (query symbol, query label)
        if self.extended:
            base += 2  # + (extra symbol, extra label)
        return base


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layer: int = 2
    n_head: int = 4
    d_mlp: int = 512
    dropout: float = 0.0
    max_seq_len: int = 64        # must be >= TaskConfig.seq_len_tokens
    tie_embeddings: bool = False # keep False: unembedding over a (symbol+label) vocab
                                  # behaves oddly if tied to input embedding for this task
    use_mlp: bool = True         # Olsson et al. (2022) / Singh et al. (2024) use ATTENTION-ONLY
                                  # transformers for induction-head studies: an MLP gives the model
                                  # a competing (partial-memorisation) pathway that can delay or
                                  # suppress induction-head formation. Set False to match that setup.


@dataclass
class TrainConfig:
    n_train: int = 50_000
    n_val: int = 2_000
    n_test: int = 2_000
    batch_size: int = 128
    n_epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    log_every: int = 50
    early_stop_patience: int = 6   # epochs with no val-loss improvement


@dataclass
class CollapseConfig:
    n_generations: int = 8
    mode: str = "pure"            # "pure" (no real data) or "mixed" (p_real injected each gen)
    p_real: float = 0.0           # fraction of each synthetic training example's LOSS
                                   # tokens that use the TRUE label instead of the model's
                                   # own sampled output. Only used when mode == "mixed".
    sample_temperature: float = 1.0  # temperature for sampling the teacher model's outputs
                                      # to build the next generation's training data.
                                      # T -> 0 is greedy/"clean" recursion, T=1 is proper
                                      # ancestral sampling ("noisy" recursion).
    retrain_from_scratch: bool = True  # if False, warm-start from the previous generation
    n_train_per_gen: int = 50_000
    seeds: tuple = (0, 1, 2)      # multiple seeds per condition for error bars
    run_ablation_every_gen: bool = True  # cheap for this model/task size; set False to save time
    n_marginal_samples: int = 5000       # samples used to estimate the model's own output marginal

    # Generation 0 needs enough training to actually form the induction circuit
    # (uses train.n_epochs / train.early_stop_patience as-is -- set those generously).
    # Generations >= 1 use THIS separate, usually much smaller, budget instead, on the
    # premise that a generation is *continuing* an already-converged model (warm
    # start) rather than re-discovering the circuit from scratch each time. If you
    # set retrain_from_scratch=True, raise these to match train.n_epochs, since each
    # generation is then training the circuit from scratch again.
    continuation_n_epochs: int = 20
    continuation_early_stop_patience: int = 8


@dataclass
class ExperimentConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    collapse: CollapseConfig = field(default_factory=CollapseConfig)
    run_name: str = "default"

    def save(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "task": asdict(self.task),
                    "model": asdict(self.model),
                    "train": asdict(self.train),
                    "collapse": asdict(self.collapse),
                    "run_name": self.run_name,
                },
                f,
                indent=2,
            )

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        return ExperimentConfig(
            task=TaskConfig(**d["task"]),
            model=ModelConfig(**d["model"]),
            train=TrainConfig(**d["train"]),
            collapse=CollapseConfig(**d["collapse"]),
            run_name=d.get("run_name", "default"),
        )
