"""Minimal prompt inference for a trained RobertaForDiffusionLM checkpoint.

Usage (Windows cmd):
  python scripts\infer_prompt.py --ckpt_dir outputs\tess_cpu_oneline_small_sc \
      --timestep 900 --simplex_value 5.0 --device cpu

If you have CUDA:
  python scripts\infer_prompt.py --ckpt_dir outputs\tess_cpu_oneline_small_sc --device cuda --timestep 900

This runs a single forward diffusion corruption (masked: only entity tokens if we can detect head/tail) and one model
prediction step, then decodes top tokens at each position. For diffusion we mimic training evaluation usage: we
prepare the simplex representation from the original tokens, add noise at a chosen timestep, and call the model.

We also print parameter counts and memory footprint estimates.
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoConfig
from sdlm.models.modeling_roberta import RobertaForDiffusionLM
from sdlm.schedulers.scheduling_simplex_ddpm import SimplexDDPMScheduler
from sdlm.utils import convert_to_simplex

def build_entity_mask(oneline_batch, tokenizer):
    """Heuristic entity mask for one-line KG format: assume format: head<TAB>relation<TAB>tail<TAB>time(optional)
    We mark tokens belonging to head and tail spans. If tabs not found, return full True mask (all tokens)."""
    masks = []
    for text in oneline_batch:
        parts = text.strip().split("\t")
        # Basic check
        if len(parts) < 3:
            encoded = tokenizer(text, add_special_tokens=True, return_attention_mask=False)
            masks.append(torch.ones(len(encoded["input_ids"]), dtype=torch.float32))
            continue
        head, rel, tail = parts[0], parts[1], parts[2]
        # Reconstruct the canonical string that tokenizer saw (rough heuristic): head\trel\ttail (ignore time)
        # We'll tokenize each part separately to know lengths.
        head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
        rel_ids = tokenizer(rel, add_special_tokens=False)["input_ids"]
        tail_ids = tokenizer(tail, add_special_tokens=False)["input_ids"]
        all_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        # Find positions: after cls (index 0). We assume layout: <s> head ... rel ... tail ... </s>
        # This may differ if additional tabs create different byte-level splits; heuristic only.
        start = 1
        head_span = range(start, start + len(head_ids))
        rel_start = start + len(head_ids)
        rel_span = range(rel_start, rel_start + len(rel_ids))
        tail_start = rel_start + len(rel_ids)
        tail_span = range(tail_start, tail_start + len(tail_ids))
        mask = torch.zeros(len(all_ids), dtype=torch.float32)
        for i in list(head_span) + list(tail_span):
            if i < len(all_ids)-1:  # avoid eos overflow
                mask[i] = 1.0
        masks.append(mask)
    # Pad to same length
    max_len = max(m.shape[0] for m in masks)
    padded = []
    for m in masks:
        if m.shape[0] < max_len:
            m = torch.cat([m, torch.zeros(max_len - m.shape[0])], dim=0)
        padded.append(m)
    return torch.stack(padded, dim=0)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Memory estimate
    fp32_bytes = total * 4
    fp16_bytes = total * 2
    def fmt(n):
        if n > 1<<30:
            return f"{n / (1<<30):.2f} GB"
        if n > 1<<20:
            return f"{n / (1<<20):.2f} MB"
        if n > 1<<10:
            return f"{n / (1<<10):.2f} KB"
        return f"{n} B"
    return {
        "total_params": total,
        "trainable_params": trainable,
        "fp32_memory": fmt(fp32_bytes),
        "fp16_memory": fmt(fp16_bytes),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="Checkpoint directory containing pytorch_model.bin & config.json")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--timestep", type=int, default=900, help="Diffusion timestep (0..num_train_timesteps-1)")
    parser.add_argument("--simplex_value", type=float, default=5.0)
    parser.add_argument("--prompts", nargs="*", default=[
        "Barack_Obama\tborn_in\tUSA\t1961",
        "Paris\tcapital_of\tFrance\t2020",
        "Einstein\tfield\tPhysics\t1955",
    ], help="Oneline KG style prompts head\trel\ttail\ttime")
    parser.add_argument("--topk", type=int, default=5, help="Top-k tokens to show per position for first sequence")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = AutoConfig.from_pretrained(args.ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir, use_fast=True)
    model = RobertaForDiffusionLM.from_pretrained(args.ckpt_dir, config=config).to(device)
    model.eval()

    # Scheduler
    scheduler = SimplexDDPMScheduler(device=device, simplex_value=args.simplex_value, num_train_timesteps=1000)

    # Tokenize prompts
    encoded = tokenizer(args.prompts, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # Build entity mask (only head & tail tokens get noise).
    entity_mask = build_entity_mask(args.prompts, tokenizer).to(device)
    # Align shape (pad mask if needed)
    if entity_mask.shape[1] < input_ids.shape[1]:
        pad_cols = input_ids.shape[1] - entity_mask.shape[1]
        entity_mask = torch.cat([entity_mask, torch.zeros(entity_mask.shape[0], pad_cols, device=device)], dim=1)
    elif entity_mask.shape[1] > input_ids.shape[1]:
        entity_mask = entity_mask[:, :input_ids.shape[1]]

    # Convert tokens to simplex representation.
    simplex = convert_to_simplex(input_ids, args.simplex_value, config.vocab_size).to(device)

    # Noise corruption at chosen timestep
    timesteps = torch.full((input_ids.shape[0],), args.timestep, dtype=torch.long, device=device)
    noise = torch.randn_like(simplex)
    noisy_simplex = scheduler.add_noise(simplex, noise, timesteps, mask=entity_mask)

    with torch.no_grad():
        out = model(
            timesteps=timesteps.float(),
            input_ids=input_ids,
            simplex=noisy_simplex,
            span_mask=entity_mask,  # reusing span_mask argument to indicate masked positions
            attention_mask=attention_mask,
            return_dict=True,
        )
    logits = out.logits  # [B, L, V]

    # For first sequence, show top-k tokens for head/tail masked positions only (entity_mask=1)
    first_logits = logits[0]
    first_ids = input_ids[0]
    first_mask = entity_mask[0].bool()
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    print("=== Parameter Stats ===")
    stats = count_parameters(model)
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\n=== Prompt 0 original tokens ===")
    print(tokenizer.convert_ids_to_tokens(first_ids.tolist()))

    print("\n=== Top-k predictions (masked entity positions) ===")
    for pos in range(first_logits.shape[0]):
        if first_mask[pos]:
            topk_vals, topk_idx = torch.topk(first_logits[pos], k=args.topk)
            decoded = [id_to_token[i.item()] for i in topk_idx]
            print(f"pos {pos}: {decoded}")

    # Reconstruct greedy sequence
    pred_ids = torch.argmax(logits, dim=-1)
    decoded_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    print("\n=== Greedy decoded texts ===")
    for i, txt in enumerate(decoded_texts):
        print(f"[{i}] {txt}")

if __name__ == "__main__":
    main()
