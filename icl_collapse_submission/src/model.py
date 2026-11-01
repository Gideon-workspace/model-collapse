"""
A small, fully explicit pre-LayerNorm causal transformer.

Written by hand (rather than via nn.TransformerEncoder / nn.MultiheadAttention)
so that:
  * per-head attention patterns are directly accessible (needed for the
    previous-token / induction-head analysis in interp.py), and
  * individual heads can be ablated (zeroed) or clamped at forward time
    (needed for the causal interventions in interp.py).

This file depends on torch. It is intentionally not imported by data.py, so
that module's logic can be unit-tested without a torch installation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, head_mask=None):
        """
        x: [B, T, D]
        head_mask: optional float tensor [n_head], multiplies each head's
            output before the final output projection (1.0 = keep, 0.0 = ablate).
        Returns:
            out: [B, T, D]
            attn: [B, n_head, T, T]  post-softmax attention weights
            head_out: [B, n_head, T, d_head]  per-head output (pre output-projection,
                      pre head_mask) -- useful for activation patching.
        """
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, n_head, T, d_head]

        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, H, T, T]
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att_scores = att_scores.masked_fill(~causal_mask, float("-inf"))
        attn = F.softmax(att_scores, dim=-1)
        attn = self.attn_dropout(attn)

        head_out = attn @ v  # [B, H, T, d_head]  (this is PRE head_mask / pre output proj)

        combined = head_out
        if head_mask is not None:
            combined = combined * head_mask.view(1, -1, 1, 1).to(x.dtype)

        combined = combined.permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.resid_dropout(self.out_proj(combined))
        return out, attn, head_out


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.fc2 = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.use_mlp = cfg.use_mlp
        if self.use_mlp:
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = MLP(cfg)

    def forward(self, x, head_mask=None):
        a_out, attn, head_out = self.attn(self.ln1(x), head_mask=head_mask)
        x = x + a_out
        if self.use_mlp:
            x = x + self.mlp(self.ln2(x))
        return x, attn, head_out


class InductionTransformer(nn.Module):
    """A GPT-style decoder-only transformer over the (symbol + label) vocabulary."""

    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, return_attn: bool = False, head_mask: dict = None):
        """
        input_ids: [B, T] long tensor of token ids.
        head_mask: optional {layer_idx: tensor[n_head]} to ablate specific heads
            in specific layers during this forward pass.
        Returns:
            logits: [B, T, vocab_size]
            attns (only if return_attn): list of [B, n_head, T, T], one per layer
            head_outs (only if return_attn): list of [B, n_head, T, d_head], one per layer
        """
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        attns, head_outs = [], []
        for i, block in enumerate(self.blocks):
            hm = head_mask.get(i) if head_mask is not None else None
            x, attn, head_out = block(x, head_mask=hm)
            if return_attn:
                attns.append(attn)
                head_outs.append(head_out)

        x = self.ln_f(x)
        logits = self.unembed(x)
        if return_attn:
            return logits, attns, head_outs
        return logits

    @torch.no_grad()
    def generate_next(self, input_ids, temperature: float = 1.0):
        """Return (sampled_token, probs) for the position AFTER the last input token.
        temperature -> 0 behaves as greedy argmax."""
        logits = self.forward(input_ids)
        last_logits = logits[:, -1, :]
        if temperature <= 1e-6:
            probs = F.softmax(last_logits, dim=-1)
            tok = torch.argmax(last_logits, dim=-1)
        else:
            probs = F.softmax(last_logits / temperature, dim=-1)
            tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return tok, probs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
