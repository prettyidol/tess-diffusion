"""KG ranking metrics: MR, MRR, Hits@k.

These utilities are adapted from the TransE-PyTorch repo but generalized to
handle either ascending scores (distance: lower is better) or descending scores
(logits/probabilities: higher is better).
"""
from __future__ import annotations

from typing import Iterable, List, Tuple
import torch


def _rank_from_scores(
    scores: torch.Tensor,
    gold_index: int,
    higher_is_better: bool,
) -> int:
    """Return 1-based rank of gold_index according to scores.

    scores: shape [N]
    gold_index: int in [0, N)
    higher_is_better: if True, sort descending; else ascending.
    """
    assert scores.dim() == 1, "scores must be 1D"
    assert 0 <= gold_index < scores.shape[0]
    if higher_is_better:
        # Larger score ranks higher
        # rank = 1 + number of items with strictly larger score
        rank = 1 + torch.sum(scores > scores[gold_index]).item()
    else:
        # Smaller score ranks higher
        rank = 1 + torch.sum(scores < scores[gold_index]).item()
    return int(rank)


def ranks_from_matrix(
    score_matrix: torch.Tensor,
    gold_indices: torch.Tensor,
    higher_is_better: bool,
) -> torch.Tensor:
    """Compute ranks for a batch.

    score_matrix: [B, N]
    gold_indices: [B] or [B,1]
    Returns: ranks [B] (int64)
    """
    if gold_indices.dim() == 2 and gold_indices.size(1) == 1:
        gold_indices = gold_indices.squeeze(1)
    assert score_matrix.dim() == 2
    assert gold_indices.dim() == 1 and gold_indices.size(0) == score_matrix.size(0)

    ranks: List[int] = []
    for i in range(score_matrix.size(0)):
        ranks.append(
            _rank_from_scores(score_matrix[i], int(gold_indices[i].item()), higher_is_better)
        )
    return torch.tensor(ranks, dtype=torch.long)


def mr(ranks: torch.Tensor) -> float:
    """Mean Rank."""
    return float(ranks.float().mean().item())


def mrr(ranks: torch.Tensor) -> float:
    """Mean Reciprocal Rank."""
    return float((1.0 / ranks.float()).mean().item())


def hits_at_k(ranks: torch.Tensor, k: int) -> float:
    """Hits@k as a fraction in [0,1]."""
    return float((ranks <= k).float().mean().item())


def summarize(ranks: torch.Tensor, ks: Iterable[int] = (1, 3, 10)) -> dict:
    """Return a dict with MR, MRR and Hits@k (as percentages)."""
    out = {
        "MR": mr(ranks),
        "MRR": mrr(ranks) * 100.0,
    }
    for k in ks:
        out[f"Hits@{k}"] = hits_at_k(ranks, k) * 100.0
    return out


# --- Compatibility helpers for TransE-PyTorch metric.py semantics ---
def transe_hit_at_k(
    predictions: torch.Tensor,
    ground_truth_idx: torch.Tensor,
    device: torch.device,
    k: int = 10,
) -> int:
    """Number of hits@k following TransE-PyTorch semantics.

    predictions: [B, N] distances (smaller is better)
    ground_truth_idx: [B] or [B,1] indices of true class
    Returns: integer count of hits in top-k w.r.t. lowest distances.
    """
    if ground_truth_idx.dim() == 2 and ground_truth_idx.size(1) == 1:
        ground_truth_idx = ground_truth_idx.squeeze(1)
    # topk with largest=False selects smallest distances
    _, indices = predictions.topk(k=k, largest=False)
    # Compare each row's top-k indices with the gold index
    hits = (indices == ground_truth_idx.unsqueeze(1)).sum().item()
    return int(hits)


def transe_mrr(
    predictions: torch.Tensor,
    ground_truth_idx: torch.Tensor,
) -> float:
    """MRR following TransE-PyTorch semantics.

    predictions: [B, N] distances (smaller is better)
    ground_truth_idx: [B] or [B,1]
    """
    if ground_truth_idx.dim() == 2 and ground_truth_idx.size(1) == 1:
        ground_truth_idx = ground_truth_idx.squeeze(1)
    # argsort ascending: column position is rank-1 (0-based)
    indices = predictions.argsort(dim=1)
    pos = (indices == ground_truth_idx.unsqueeze(1)).nonzero()
    if pos.numel() == 0:
        return 0.0
    # pos shape [M, 2]: (row_idx, col_idx). We want col_idx as 0-based rank position.
    ranks0 = pos[:, 1].float()
    recip = (1.0 / (ranks0 + 1.0)).sum().item()
    return float(recip)
