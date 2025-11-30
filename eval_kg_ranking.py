"""Evaluate KG ranking metrics (MR, MRR, Hits@k) on oneline TESS data.

Usage (Windows cmd):
  python eval_kg_ranking.py --test_file tess_test1_oneline.txt --mode tail --k 1 3 10

This script provides two simple scorers for now:
  - random: random ranking baseline
  - freq: rank by (h,r,*) tail frequency estimated from the same file

It also exposes a hook `score_candidates_with_model` to integrate TESS logits
later if needed.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple
from bisect import bisect_left, bisect_right

import torch

from sdlm.metrics.kg_metrics import ranks_from_matrix, summarize
from sdlm.data.oneline_kg_parser import parse_oneline_triples, parse_oneline_quads, build_id_mappings
from sdlm.models.modeling_roberta import RobertaForDiffusionLM
from sdlm.schedulers.scheduling_simplex_ddpm import SimplexDDPMScheduler
from sdlm.utils import convert_to_simplex

# Optional model scorer
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from torch.cuda.amp import autocast
except Exception:
    autocast = None


Triple = Tuple[str, str, str]
Quad = Tuple[str, str, str, str]


def _build_candidate_lists(
    triples: List[Triple], mode: str
) -> Tuple[Dict[Tuple[str, str], List[str]], List[int]]:
    """Build candidate entities per (h,r) or (r,t), and gold indices per query.

    Returns:
      candidates: mapping from query key to list of unique candidate entities
      gold_indices: indices into that list pointing to the true entity per query
    """
    assert mode in {"tail", "head"}

    # collect candidates
    cand: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    gold: List[int] = []
    keys: List[Tuple[str, str]] = []

    # maintain a set for uniqueness per key
    seen: Dict[Tuple[str, str], set] = defaultdict(set)

    for h, r, t in triples:
        if mode == "tail":
            key = (h, r)
            ent = t
        else:
            key = (r, t)
            ent = h
        if ent not in seen[key]:
            seen[key].add(ent)
            cand[key].append(ent)
        keys.append(key)
        gold.append(cand[key].index(ent))

    return cand, gold


def _build_candidate_lists_with_time(
    quads: List[Quad], mode: str, compare: str = "lt"
) -> Tuple[Dict[Tuple[str, str, str], List[str]], List[int], List[Tuple[str, str, str]]]:
    """Build time-aware candidates and gold using query time.

    compare:
      - 'lt': only facts strictly earlier than query time
      - 'le': allow same day (<=)
    Keys:
      - tail mode: (h, r, time_q) -> candidates are tails with time <(=) time_q
      - head mode: (r, t, time_q) -> candidates are heads with time <(=) time_q
    """
    assert mode in {"tail", "head"}
    assert compare in {"lt", "le"}

    def ok(t_fact: str, t_query: str) -> bool:
        if not t_query:
            return True
        if not t_fact:
            return True
        return t_fact < t_query if compare == "lt" else t_fact <= t_query

    cand: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    gold: List[int] = []
    keys: List[Tuple[str, str, str]] = []
    seen: Dict[Tuple[str, str, str], set] = defaultdict(set)

    for h, r, t, time_q in quads:
        if mode == "tail":
            key = (h, r, time_q)
            ent = t
        else:
            key = (r, t, time_q)
            ent = h

        # collect from all quads that match (h,r,*) or (*,r,t) and satisfy time condition
        for hh, rr, tt, tf in quads:
            if mode == "tail" and hh == h and rr == r and ok(tf, time_q):
                if tt not in seen[key]:
                    seen[key].add(tt)
                    cand[key].append(tt)
            elif mode == "head" and rr == r and tt == t and ok(tf, time_q):
                if hh not in seen[key]:
                    seen[key].add(hh)
                    cand[key].append(hh)

        keys.append(key)
        # ensure gold is included
        if ent not in seen[key]:
            seen[key].add(ent)
            cand[key].append(ent)
        gold.append(cand[key].index(ent))

    return cand, gold, keys


def _build_time_index(
    quads: List[Quad], mode: str
) -> Dict[Tuple[str, str], Dict[str, List[str]]]:
    """Build index: group_key -> {entity: sorted list of times}.

    - tail mode groups by (h, r) and indexes tails t by time
    - head mode groups by (r, t) and indexes heads h by time
    """
    idx: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    if mode == "tail":
        for h, r, t, tf in quads:
            idx[(h, r)][t].append(tf)
    else:
        for h, r, t, tf in quads:
            idx[(r, t)][h].append(tf)
    # sort times per entity
    for _, ents in idx.items():
        for ent, times in ents.items():
            times.sort()
    return idx


def _build_candidates_with_index(
    quads: List[Quad], mode: str, compare: str
) -> Tuple[
    Dict[Tuple[str, str, str], List[str]],
    List[int],
    List[Tuple[str, str, str]],
    Dict[Tuple[str, str, str], Counter],
    Dict[Tuple[str, str, str], Counter],
]:
    """Use time index and binary search to build candidates and gold efficiently.

    Returns candidates_map, gold_indices, keys, tail_counter, head_counter.
    Only one of tail_counter/head_counter will be filled depending on mode.
    """
    assert mode in {"tail", "head"}
    use_right = compare == "le"
    time_index = _build_time_index(quads, mode)

    candidates_map: Dict[Tuple[str, str, str], List[str]] = {}
    gold_indices: List[int] = []
    keys: List[Tuple[str, str, str]] = []
    tail_counter: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    head_counter: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)

    for h, r, t, tq in quads:
        if mode == "tail":
            group_key = (h, r)
            key = (h, r, tq)
            ents_times = time_index.get(group_key, {})
            # compute counts up to tq for each entity in the group via bisect
            cnt = Counter()
            for ent, times in ents_times.items():
                k = bisect_right(times, tq) if use_right else bisect_left(times, tq)
                if k > 0:
                    cnt[ent] = k
            # build candidate list preserving a deterministic order (by descending count then name)
            if cnt:
                # sort by count desc, then ent name for stability
                sorted_ents = sorted(cnt.keys(), key=lambda e: (-cnt[e], e))
            else:
                sorted_ents = []
            # ensure gold entity exists
            if t not in sorted_ents:
                sorted_ents.append(t)
                cnt[t] += 0
            candidates_map[key] = sorted_ents
            keys.append(key)
            gold_indices.append(sorted_ents.index(t))
            tail_counter[key] = cnt
        else:
            group_key = (r, t)
            key = (r, t, tq)
            ents_times = time_index.get(group_key, {})
            cnt = Counter()
            for ent, times in ents_times.items():
                k = bisect_right(times, tq) if use_right else bisect_left(times, tq)
                if k > 0:
                    cnt[ent] = k
            if cnt:
                sorted_ents = sorted(cnt.keys(), key=lambda e: (-cnt[e], e))
            else:
                sorted_ents = []
            if h not in sorted_ents:
                sorted_ents.append(h)
                cnt[h] += 0
            candidates_map[key] = sorted_ents
            keys.append(key)
            gold_indices.append(sorted_ents.index(h))
            head_counter[key] = cnt

    return candidates_map, gold_indices, keys, tail_counter, head_counter


def _build_facts_sets(quads: List[Quad]) -> Tuple[set, Dict[Tuple[str, str], Dict[str, List[str]]]]:
    """Build a set of triples (h,r,t) ignoring time and a time index for filtered checks."""
    triple_set = set()
    for h, r, t, _ in quads:
        triple_set.add((h, r, t))
    # time index for both modes: we'll build two for convenience
    # but reusing builder per mode when filtering time-aware
    return triple_set, _build_time_index(quads, mode="tail")


def _build_sampled_candidates(
    quads: List[Quad],
    all_entities: List[str],
    mode: str,
    neg_k: int,
    filtered: bool,
    time_compare: str,
) -> Tuple[Dict[Tuple[str, str, str], List[str]], List[int], List[Tuple[str, str, str]]]:
    """Per TransE negative sampling, build candidate list = [gold + K negatives].

    - filtered=True: remove negatives that correspond to true facts (raw vs filtered)
    - time-aware filtering: if filtered, we remove negatives that were true strictly earlier (lt)或(<=)查询时间
    """
    assert mode in {"tail", "head"}
    use_right = time_compare == "le"
    # sets for filtering
    triple_set = {(h, r, t) for h, r, t, _ in quads}
    # time index per mode for time-aware filtering
    time_index = _build_time_index(quads, mode)

    candidates_map: Dict[Tuple[str, str, str], List[str]] = {}
    gold_indices: List[int] = []
    keys: List[Tuple[str, str, str]] = []

    all_set = set(all_entities)

    for h, r, t, tq in quads:
        if mode == "tail":
            group_key = (h, r)
            gold = t
        else:
            group_key = (r, t)
            gold = h

        # Build forbidden set if filtered
        forbidden = set([gold])
        if filtered:
            ents_times = time_index.get(group_key, {})
            if mode == "tail":
                # forbid tails that have been seen with (h,r) before tq
                for ent, times in ents_times.items():
                    k = bisect_right(times, tq) if use_right else bisect_left(times, tq)
                    if k > 0:
                        forbidden.add(ent)
            else:
                # forbid heads seen with (r,t) before tq
                for ent, times in ents_times.items():
                    k = bisect_right(times, tq) if use_right else bisect_left(times, tq)
                    if k > 0:
                        forbidden.add(ent)

        pool = list(all_set - forbidden)
        # If pool is empty, we end up with only gold
        if neg_k > 0 and len(pool) > 0:
            k = min(neg_k, len(pool))
            negs = random.sample(pool, k)
        else:
            negs = []

        if mode == "tail":
            cands = [gold] + negs
            key = (h, r, tq)
            gold_idx = 0
        else:
            cands = [gold] + negs
            key = (r, t, tq)
            gold_idx = 0

        candidates_map[key] = cands
        keys.append(key)
        gold_indices.append(gold_idx)

    return candidates_map, gold_indices, keys


def _random_scores(num_candidates: int) -> torch.Tensor:
    return torch.rand(num_candidates)


def _freq_scores(candidates: Sequence[str], counter: Counter) -> torch.Tensor:
    vals = [counter[c] for c in candidates]
    return torch.tensor(vals, dtype=torch.float32)


def _model_score_batch(
    model,
    tokenizer,
    device: torch.device,
    prefix: str,
    candidates: Sequence[str],
    batch_size: int = 16,
) -> torch.Tensor:
    """Score candidates by conditional log-probability using a causal LM.

    Returns a 1D tensor of summed log-probabilities (higher is better).
    We compute p(candidate_tokens | prefix_tokens) using the causal model.
    """
    scores = []
    model.eval()
    with torch.no_grad():
        # Pre-tokenize prefix once
        prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        prefix_len = prefix_ids.size(1)

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            # Build concatenated inputs: prefix + candidate for each candidate
            input_ids_list = []
            cand_lens = []
            for cand in batch:
                # no special tokens to keep consistent token positions
                cand_ids = tokenizer(cand, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
                if cand_ids.numel() == 0:
                    # fallback: unknown token (score very low)
                    input_ids = prefix_ids.squeeze(0).clone()
                    cand_lens.append(0)
                else:
                    input_ids = torch.cat([prefix_ids.squeeze(0), cand_ids], dim=0)
                    cand_lens.append(cand_ids.size(0))
                input_ids_list.append(input_ids)

            # pad batch
            max_len = max(x.size(0) for x in input_ids_list)
            input_batch = torch.stack(
                [torch.cat([x, x.new_full((max_len - x.size(0),), tokenizer.pad_token_id)]) for x in input_ids_list]
            ).to(device)

            outputs = model(input_batch)[0]  # logits [B, L, V]
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)

            for b_idx, cand_len in enumerate(cand_lens):
                if cand_len == 0:
                    scores.append(float("-1e9"))
                    continue
                # candidate token positions are prefix_len .. prefix_len + cand_len - 1
                start = prefix_len
                end = prefix_len + cand_len
                # gather log probs of the actual tokens
                ids = input_batch[b_idx, start:end]
                token_logps = log_probs[b_idx, start:end].gather(1, ids.unsqueeze(1)).squeeze(1)
                scores.append(float(token_logps.sum().cpu().item()))

    return torch.tensor(scores, dtype=torch.float32)


def create_model_scorer(model_name_or_path: str, device: str, mode: str):
    device_t = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device_t)

    def scorer(candidates: Sequence[str], query_key: Tuple[str, str, str]) -> torch.Tensor:
        # query_key is (h, r, time) for tail mode or (r, t, time) for head mode
        if mode == "tail":
            h, r, _ = query_key
            prefix = f"{h}\t{r}\t"
        else:
            # head mode: we condition on relation+tail and score candidate heads
            r, t, _ = query_key
            prefix = f"{r}\t{t}\t"
        return _model_score_batch(model, tokenizer, device_t, prefix, candidates, batch_size=16)

    return scorer


def _pad_and_stack(tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    """Pad 1D tensors to the same length and stack into [B, L]."""
    if len(tensors) == 0:
        return torch.empty(0, dtype=torch.long)
    max_len = max(t.size(0) for t in tensors)
    padded = [torch.cat([t, t.new_full((max_len - t.size(0),), pad_value)]) for t in tensors]
    return torch.stack(padded, dim=0)


def create_tess_scorer(
    model_name_or_path: str,
    device: str,
    mode: str,
    tess_simplex_value: float = 5.0,
    tess_beta_schedule: str = "squaredcos_cap_v2",
    tess_num_steps: int = 1000,
    tess_t_eval: int = 200,
    batch_size: int = 16,
    amp: bool = False,
    cache_entities: bool = True,
):
    """Create a non-autoregressive scorer using TESS diffusion LM.

    For each candidate, we construct an input where the candidate entity span is masked
    (span_mask=True on that span), then compute the model's masked LM loss under a fixed
    noise level (one-step denoising objective). We return negative loss as the score.
    """
    device_t = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = RobertaForDiffusionLM.from_pretrained(model_name_or_path).to(device_t)
    model.eval()

    # Build an inference scheduler only to create a consistent noisy simplex at a fixed timestep
    scheduler = SimplexDDPMScheduler(
        device=device_t,
        simplex_value=tess_simplex_value,
        num_train_timesteps=tess_num_steps,
        num_inference_timesteps=tess_num_steps,
        beta_schedule=tess_beta_schedule,
        clip_sample=False,
    )
    # clamp eval timestep to range
    t_eval = int(max(0, min(tess_num_steps - 1, tess_t_eval)))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    token_cache: Dict[str, torch.Tensor] = {} if cache_entities else None

    def build_inputs(prefix: str, cand_list: Sequence[str]):
        # tokenize prefix once
        prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        input_ids_list: List[torch.Tensor] = []
        span_masks: List[torch.Tensor] = []
        for cand in cand_list:
            if token_cache is not None and cand in token_cache:
                cand_ids = token_cache[cand]
            else:
                cand_ids = tokenizer(cand, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
                if token_cache is not None:
                    token_cache[cand] = cand_ids
            if cand_ids.numel() == 0:
                # handle empty candidate by using a single pad, but mark span length 0 later
                ids = prefix_ids.clone()
                mask = torch.zeros_like(prefix_ids, dtype=torch.bool)
            else:
                ids = torch.cat([prefix_ids, cand_ids], dim=0)
                mask = torch.cat([torch.zeros_like(prefix_ids, dtype=torch.bool), torch.ones_like(cand_ids, dtype=torch.bool)], dim=0)
            input_ids_list.append(ids)
            span_masks.append(mask)
        # add special tokens (<s> ... </s>)
        input_ids_list = [tokenizer.build_inputs_with_special_tokens(seq.tolist()) for seq in input_ids_list]
        # rebuild tensors and corresponding span masks (shift by one due to bos)
        input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in input_ids_list]
        span_masks = [torch.cat([torch.tensor([False], dtype=torch.bool), m, torch.tensor([False], dtype=torch.bool)]) for m in span_masks]

        input_ids = _pad_and_stack(input_ids_list, pad_id)
        span_mask = _pad_and_stack(span_masks, 0)
        return input_ids.to(device_t), span_mask.to(device_t)

    @torch.no_grad()
    def scorer(candidates: Sequence[str], query_key: Tuple[str, str, str]) -> torch.Tensor:
        # Build prefix text depending on ranking mode
        if mode == "tail":
            h, r, _ = query_key
            prefix = f"{h}\t{r}\t"
        else:
            r, t, _ = query_key
            prefix = f"{r}\t{t}\t"

        scores: List[float] = []
        for i in range(0, len(candidates), batch_size):
            batch_cands = candidates[i : i + batch_size]
            input_ids, span_mask = build_inputs(prefix, batch_cands)
            # Build simplex and add noise at fixed timestep
            simplex = convert_to_simplex(input_ids, tess_simplex_value, model.config.vocab_size)
            noise = tess_simplex_value * torch.randn_like(simplex, dtype=simplex.dtype, device=simplex.device)
            timesteps = torch.full((input_ids.size(0),), fill_value=t_eval, dtype=torch.int64, device=device_t)
            # 仅对候选实体 span 加噪，关系与时间保持原样
            noisy_simplex = scheduler.add_noise(simplex, noise, timesteps, mask=span_mask)
            # scale timesteps as in training
            timesteps_scaled = timesteps.float() / float(len(scheduler))
            if amp and device_t.type == "cuda" and autocast is not None:
                with autocast(dtype=torch.float16):
                    outputs = model(
                        timesteps=timesteps_scaled,
                        input_ids=input_ids,
                        simplex=noisy_simplex,
                        span_mask=span_mask,
                        return_dict=True,
                    )
            else:
                outputs = model(
                    timesteps=timesteps_scaled,
                    input_ids=input_ids,
                    simplex=noisy_simplex,
                    span_mask=span_mask,
                    return_dict=True,
                )
            # Compute per-sample masked LM loss manually from logits and labels
            logits = outputs.logits  # [B, L, V]
            labels = input_ids.clone()
            labels[~span_mask] = -100  # ignore unmasked tokens
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            B, L, V = log_probs.shape
            # gather log-prob of true labels for masked positions
            per_sample_losses: List[float] = []
            for b in range(B):
                mask_b = span_mask[b]  # [L]
                if mask_b.sum().item() == 0:
                    per_sample_losses.append(float("inf"))  # no masked tokens -> worst loss
                    continue
                ids_b = labels[b, mask_b]  # [M]
                lp_b = log_probs[b, mask_b, :]  # [M, V]
                token_lp = lp_b.gather(1, ids_b.unsqueeze(1)).squeeze(1)  # [M]
                # negative mean log-prob as loss
                loss_b = -token_lp.mean()
                per_sample_losses.append(loss_b.detach().float().item())
            # Score is negative loss
            scores.extend([-x if x != float("inf") else -1e9 for x in per_sample_losses])
        return torch.tensor(scores, dtype=torch.float32)

    return scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", type=str, default="tess_test1_oneline.txt")
    ap.add_argument("--mode", type=str, default="tail", choices=["head", "tail"], help="Rank head or tail")
    ap.add_argument(
        "--scorer",
        type=str,
        default="freq",
        choices=["random", "freq", "model", "tess"],
        help="Scoring function: random/freq baselines, model (causal LM), or tess (diffusion LM)",
    )
    ap.add_argument("--k", type=int, nargs="*", default=[1, 3, 10])
    ap.add_argument("--time_compare", type=str, default="lt", choices=["lt", "le"], help="Time filter: lt or le")
    ap.add_argument(
        "--candidates",
        type=str,
        default="observed",
        choices=["observed", "all", "sampled"],
        help="Candidate set per query: observed (time-aware), all entities, or sampled negatives (TransE-style)",
    )
    ap.add_argument("--model_name_or_path", type=str, default=None, help="Model path: causal LM for --scorer model, or TESS checkpoint for --scorer tess")
    ap.add_argument("--device", type=str, default="cpu", help="Device for model (cpu or cuda)")
    # TESS-specific knobs
    ap.add_argument("--tess_simplex_value", type=float, default=5.0, help="TESS simplex value used in training")
    ap.add_argument(
        "--tess_beta_schedule",
        type=str,
        default="squaredcos_cap_v2",
        choices=["linear", "scaled_linear", "squaredcos_cap_v2", "squaredcos_improved_ddpm", "sigmoid"],
        help="Scheduler beta schedule used in training",
    )
    ap.add_argument("--tess_num_steps", type=int, default=500, help="Number of diffusion steps used in training (should match config)")
    ap.add_argument("--tess_t_eval", type=int, default=60, help="Fixed timestep for TESS evaluation (recommended 40-80 for optimal results)")
    ap.add_argument("--model_batch_size", type=int, default=16, help="Batch size when scoring with a model")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (fp16) on CUDA for faster TESS scoring")
    ap.add_argument("--cache_entities", action="store_true", help="Cache tokenization for candidate entities to reduce CPU overhead")
    # Negative sampling options (TransE-style)
    ap.add_argument("--neg_k", type=int, default=128, help="Number of negatives per query when --candidates sampled (recommended 64-256 for optimal coverage)")
    ap.add_argument("--filtered", action="store_true", help="Filtered setting: remove negatives that correspond to true facts (time-aware)")
    # Quick sampling evaluation and reproducibility
    ap.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="If >0, randomly subsample at most this many queries for a quick evaluation",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling/negatives")
    ap.add_argument("--start_from", type=int, default=0, help="Start index for sequential slicing (use with --max_queries) for sharding or resume")
    args = ap.parse_args()

    # set random seeds for reproducible sampling and negative sampling
    random.seed(args.seed)
    try:
        torch.manual_seed(args.seed)
    except Exception:
        pass

    quads = parse_oneline_quads(args.test_file)
    if len(quads) == 0:
        print("No quads parsed from", args.test_file)
        return

    # Build global entity list
    all_entities = sorted({h for h, _, _, _ in quads} | {t for _, _, t, _ in quads})

    if args.candidates == "observed":
        # 高效构建：利用按键+时间的倒排索引与二分
        (
            candidates_map,
            gold_indices,
            keys,
            tail_counter,
            head_counter,
        ) = _build_candidates_with_index(quads, mode=args.mode, compare=args.time_compare)
    else:
        if args.candidates == "all":
            # TransE-style: all entities as candidates, ignore time filtering for candidates
            keys = []
            gold_indices = []
            candidates_map = {}

            # for freq scorer, we still compute per-time key counts but scores for unseen entities are 0
            tail_counter = defaultdict(Counter)
            head_counter = defaultdict(Counter)
            for h, r, t, tf in quads:
                tail_counter[(h, r, tf)].update([t])
                head_counter[(r, t, tf)].update([h])

            for h, r, t, tf in quads:
                if args.mode == "tail":
                    key = (h, r, tf)
                    true_ent = t
                else:
                    key = (r, t, tf)
                    true_ent = h
                candidates_map[key] = all_entities  # shared reference, not modifying
                keys.append(key)
                gold_indices.append(all_entities.index(true_ent))
        else:
            # Negative-sampled candidates per TransE paper
            candidates_map, gold_indices, keys = _build_sampled_candidates(
                quads=quads,
                all_entities=all_entities,
                mode=args.mode,
                neg_k=args.neg_k,
                filtered=args.filtered,
                time_compare=args.time_compare,
            )
            # freq counters are not meaningful here; create empty ones
            tail_counter = defaultdict(Counter)
            head_counter = defaultdict(Counter)

    # Optionally subsample queries for quick evaluation
    n_all = len(keys)
    # Priority: if start_from > 0 (or provided), do deterministic sequential slicing for sharding/resume
    if (args.start_from and args.start_from > 0) or (args.max_queries and args.max_queries > 0):
        start = max(0, min(n_all, args.start_from))
        end = n_all if not args.max_queries or args.max_queries <= 0 else min(n_all, start + args.max_queries)
        if start < end:
            keys = keys[start:end]
            gold_indices = gold_indices[start:end]
            candidates_map = {k: candidates_map[k] for k in keys}
            if args.scorer == "freq":
                if args.mode == "tail":
                    tail_counter = defaultdict(Counter, {k: tail_counter[k] for k in keys})
                else:
                    head_counter = defaultdict(Counter, {k: head_counter[k] for k in keys})
        # If start_from==0 and max_queries==0: no slicing

    # build score matrix
    scores: List[torch.Tensor] = []
    gold = torch.tensor(gold_indices, dtype=torch.long)
    model_scorer = None
    if args.scorer in ("model", "tess"):
        if args.model_name_or_path is None:
            raise ValueError("--model_name_or_path is required when --scorer model/tess")
        if args.scorer == "model":
            model_scorer = create_model_scorer(args.model_name_or_path, args.device, args.mode)
        else:
            model_scorer = create_tess_scorer(
                model_name_or_path=args.model_name_or_path,
                device=args.device,
                mode=args.mode,
                tess_simplex_value=args.tess_simplex_value,
                tess_beta_schedule=args.tess_beta_schedule,
                tess_num_steps=args.tess_num_steps,
                tess_t_eval=args.tess_t_eval,
                batch_size=args.model_batch_size,
                amp=args.amp,
                cache_entities=args.cache_entities,
            )
    for key in keys:
        cands = candidates_map[key]
        if args.scorer == "random":
            s = _random_scores(len(cands))
        elif args.scorer == "freq":
            counter = tail_counter[key] if args.mode == "tail" else head_counter[key]
            s = _freq_scores(cands, counter)
        else:
            # model-based scorers
            s = model_scorer(cands, key)
        scores.append(s)

    # pad to same length for a matrix
    max_len = max(s.numel() for s in scores)
    score_mat = []
    for s in scores:
        if s.numel() < max_len:
            pad = torch.full((max_len - s.numel(),), fill_value=float("-inf"))
            s = torch.cat([s, pad], dim=0)
        score_mat.append(s.unsqueeze(0))
    score_mat = torch.cat(score_mat, dim=0)

    ranks = ranks_from_matrix(score_mat, gold, higher_is_better=True)
    report = summarize(ranks, ks=args.k)

    print({**report, "count": len(keys), "mode": args.mode, "scorer": args.scorer})


if __name__ == "__main__":
    main()
