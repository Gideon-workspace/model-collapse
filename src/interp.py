"""
Interpretability toolkit: attention-based circuit scores (previous-token /
induction scores), head ablation, causal activation patching, and symbol
embedding geometry. Used to identify and track the induction circuit
(Fig. 1 of the brief) across training and across model-collapse generations.

References the methodology of Olsson et al. (2022) and Conmy et al. (2023).
"""

import numpy as np
import torch

from torch_data import masked_cross_entropy
from metrics import pca_2d, participation_ratio, mean_pairwise_cosine_similarity


# --------------------------------------------------------------------------
# Dataset-level evaluation with optional head ablation
# --------------------------------------------------------------------------

def build_head_mask(n_layer, n_head, ablate=None, keep_only=None, device="cpu"):
    """ablate: list of (layer, head) tuples to zero out.
    keep_only: (layer, head) tuple -- zero every OTHER head in that layer only
    (heads in other layers are left untouched)."""
    head_mask = {l: torch.ones(n_head, device=device) for l in range(n_layer)}
    if ablate:
        for (l, h) in ablate:
            head_mask[l][h] = 0.0
    if keep_only is not None:
        l, h = keep_only
        mask = torch.zeros(n_head, device=device)
        mask[h] = 1.0
        head_mask[l] = mask
    return head_mask


@torch.no_grad()
def evaluate_dataset(model, dataset, device, head_mask=None, batch_size=256):
    model.eval()
    N = len(dataset)
    tot_loss, tot_correct, tot_count = 0.0, 0, 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        input_ids = dataset.input_ids[start:end].to(device)
        targets = dataset.targets[start:end].to(device)
        loss_mask = dataset.loss_mask[start:end].to(device)
        logits = model(input_ids, head_mask=head_mask)
        loss = masked_cross_entropy(logits, targets, loss_mask)
        preds = logits.argmax(dim=-1)
        correct = ((preds == targets) & loss_mask).sum().item()
        count = loss_mask.sum().item()
        tot_loss += loss.item() * count
        tot_correct += correct
        tot_count += count
    return dict(loss=tot_loss / max(1, tot_count), accuracy=tot_correct / max(1, tot_count))


def run_ablation_table(model, dataset, device, batch_size=256):
    """Single-head ablation and keep-only ablation over every (layer, head).
    Returns a list of dict rows, ready to dump to JSON / a pandas DataFrame."""
    n_layer, n_head = model.cfg.n_layer, model.cfg.n_head
    baseline = evaluate_dataset(model, dataset, device, head_mask=None, batch_size=batch_size)
    rows = [dict(type="baseline", layer=None, head=None, accuracy=baseline["accuracy"],
                 loss=baseline["loss"], delta_acc=0.0)]

    for l in range(n_layer):
        for h in range(n_head):
            hm = build_head_mask(n_layer, n_head, ablate=[(l, h)], device=device)
            r = evaluate_dataset(model, dataset, device, head_mask=hm, batch_size=batch_size)
            rows.append(dict(type="single_ablation", layer=l, head=h, accuracy=r["accuracy"],
                              loss=r["loss"], delta_acc=r["accuracy"] - baseline["accuracy"]))

    for l in range(n_layer):
        for h in range(n_head):
            hm = build_head_mask(n_layer, n_head, keep_only=(l, h), device=device)
            r = evaluate_dataset(model, dataset, device, head_mask=hm, batch_size=batch_size)
            rows.append(dict(type="keep_only", layer=l, head=h, accuracy=r["accuracy"],
                              loss=r["loss"], delta_acc=r["accuracy"] - baseline["accuracy"]))

    return rows


# --------------------------------------------------------------------------
# Attention-based circuit scores
# --------------------------------------------------------------------------

@torch.no_grad()
def collect_attention_stats(model, dataset, device, batch_size=256):
    """Previous-token score and induction score, averaged over the dataset,
    for every (layer, head).

    previous_token_score[l, h]: mean attention weight from position t to t-1,
        averaged over all t and all examples. High for a "previous-token head".
    induction_score[l, h]: mean attention weight from the query-symbol position
        to the label position that immediately follows the most recent matching
        context symbol. High for the/an "induction head".
    """
    model.eval()
    n_layer, n_head = model.cfg.n_layer, model.cfg.n_head
    N = len(dataset)
    qpos = dataset.positions["query_symbol_pos"]

    prev_sum = np.zeros((n_layer, n_head))
    prev_denom = 0
    ind_sum = np.zeros((n_layer, n_head))
    ind_denom = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        input_ids = dataset.input_ids[start:end].to(device)
        B, T = input_ids.shape
        _, attns, _ = model(input_ids, return_attn=True)

        last_match = dataset.last_match_pos[start:end]           # [B] position within ctx
        src_pos = 2 * last_match + 1                              # [B] token index of label to copy
        src_pos_t = torch.from_numpy(src_pos).long().to(device)

        for l in range(n_layer):
            att = attns[l]  # [B, H, T, T]

            # previous-token score
            t_idx = torch.arange(1, T, device=device)
            pt = att[:, :, t_idx, t_idx - 1]  # [B, H, T-1]
            prev_sum[l] += pt.sum(dim=(0, 2)).cpu().numpy()

            # induction score
            att_at_q = att[:, :, qpos, :]  # [B, H, T]
            idx = src_pos_t.view(B, 1, 1).expand(B, n_head, 1)
            ind_vals = torch.gather(att_at_q, 2, idx).squeeze(-1)  # [B, H]
            ind_sum[l] += ind_vals.sum(dim=0).cpu().numpy()

        prev_denom += B * (T - 1)
        ind_denom += B

    return dict(
        previous_token_score=prev_sum / max(1, prev_denom),
        induction_score=ind_sum / max(1, ind_denom),
    )


# --------------------------------------------------------------------------
# Causal activation patching (Conmy et al., 2023 style)
# --------------------------------------------------------------------------

def _patch_hook_factory(head_idx, pos_idx, replacement):
    def hook(module, inputs, output):
        out, attn, head_out = output
        head_out = head_out.clone()
        B = head_out.shape[0]
        head_out[torch.arange(B, device=head_out.device), head_idx, pos_idx, :] = replacement
        Bh, H, T, dh = head_out.shape
        combined = head_out.permute(0, 2, 1, 3).reshape(Bh, T, H * dh)
        new_out = module.resid_dropout(module.out_proj(combined))
        return new_out, attn, head_out
    return hook


@torch.no_grad()
def run_activation_patching(model, dataset, device, layer, head, n_pairs=200, seed=0, batch_size=256):
    """Patch `head`'s output in `layer`, at the query-symbol position, from a
    batch of 'clean' examples into a batch of 'corrupted' examples (same
    template, different content), and measure whether the corrupted example's
    prediction flips to the clean example's answer. A high flip rate is causal
    evidence that this head alone carries the copied label to the query position.
    """
    model.eval()
    n_layer, n_head = model.cfg.n_layer, model.cfg.n_head
    qpos = dataset.positions["query_symbol_pos"]
    N = len(dataset)

    rng = np.random.default_rng(seed)
    idx_clean = rng.choice(N, size=n_pairs, replace=False)
    idx_corrupt = rng.choice(N, size=n_pairs, replace=False)

    clean_target = dataset.targets[idx_clean, qpos]
    corrupt_target = dataset.targets[idx_corrupt, qpos]
    keep = (clean_target != corrupt_target).numpy()
    idx_clean, idx_corrupt = idx_clean[keep], idx_corrupt[keep]
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise RuntimeError("No clean/corrupt pairs with differing answers were found; increase n_pairs.")

    clean_input = dataset.input_ids[idx_clean].to(device)
    corrupt_input = dataset.input_ids[idx_corrupt].to(device)
    clean_target = dataset.targets[idx_clean, qpos].to(device)
    corrupt_target = dataset.targets[idx_corrupt, qpos].to(device)

    _, _, clean_head_outs = model(clean_input, return_attn=True)
    donor = clean_head_outs[layer][:, head, qpos, :]  # [n_kept, d_head]

    corrupt_logits_orig = model(corrupt_input)
    corrupt_pred_orig = corrupt_logits_orig[:, qpos, :].argmax(-1)
    baseline_own_acc = (corrupt_pred_orig == corrupt_target).float().mean().item()

    attn_module = model.blocks[layer].attn
    pos_idx = torch.full((n_kept,), qpos, device=device, dtype=torch.long)
    handle = attn_module.register_forward_hook(_patch_hook_factory(head, pos_idx, donor))
    try:
        patched_logits = model(corrupt_input)
    finally:
        handle.remove()
    patched_pred = patched_logits[:, qpos, :].argmax(-1)

    flip_to_clean_rate = (patched_pred == clean_target).float().mean().item()
    still_own_rate = (patched_pred == corrupt_target).float().mean().item()

    return dict(
        layer=layer, head=head, n_pairs_used=n_kept,
        baseline_corrupt_own_acc=baseline_own_acc,
        patched_flip_to_clean_rate=flip_to_clean_rate,
        patched_still_own_rate=still_own_rate,
    )


# --------------------------------------------------------------------------
# Embedding geometry
# --------------------------------------------------------------------------

def symbol_embedding_geometry(model, task, symbol_ids=None):
    """PCA / anisotropy / effective-dimensionality of the learned symbol
    embeddings. Useful to track representation collapse across generations."""
    W = model.tok_emb.weight.detach().cpu().numpy()
    sym_emb = W[: task.n_symbols]
    if symbol_ids is not None:
        sym_emb = sym_emb[np.asarray(symbol_ids)]
    X2d, explained, _ = pca_2d(sym_emb)
    return dict(
        X2d=X2d,
        explained_var_ratio=explained,
        participation_ratio=participation_ratio(sym_emb),
        mean_cosine_similarity=mean_pairwise_cosine_similarity(sym_emb),
        embeddings=sym_emb,
    )
