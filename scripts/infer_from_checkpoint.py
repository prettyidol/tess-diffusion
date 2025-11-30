#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple inference script for Diffusion LM checkpoint.

Usage:
    python scripts/infer_from_checkpoint.py \
        --ckpt_dir outputs/tess_cpu_oneline_small_sc \
        --prompts "今天的天气很好" "人工智能正在改变世界" "写一首关于秋天的短诗" \
        --steps 5 \
        --top_p 0.9 \
        --temperature 1.0

The diffusion model here is trained to denoise simplex logits rather than doing standard autoregressive generation.
For a quick qualitative check we implement an extremely simplified iterative denoising loop:
    1. Start from the input prompt token ids (or an empty prompt) and convert to simplex.
    2. At each step pick a random timestep scalar (simulating noise level) and call model.forward with previous simplex.
    3. Sample next simplex via nucleus sampling on logits (using code adapted from inference_utils.logits_projection).
This is NOT the full sampling procedure used in the paper (which employs a noise scheduler); it is a lightweight demo.

We also print parameter counts and basic config stats.
"""
import os
import math
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from sdlm.models import RobertaForDiffusionLM
from sdlm.inference.inference_utils import sample_logits
from sdlm.utils import convert_to_simplex


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_model_and_tokenizer(ckpt_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    # We need to pass config + weights manually because this is a custom class.
    model = RobertaForDiffusionLM.from_pretrained(ckpt_dir)
    model.eval().to(device)
    return model, tokenizer


def _compute_tail_char_span(prompt: str):
    """Return (start, end) char offsets for the 3rd TAB-separated field (tail). If not found, return None."""
    parts = prompt.split("\t")
    if len(parts) < 3:
        return None
    before = "\t".join(parts[:2]) + "\t"
    start = len(before)
    end = start + len(parts[2])
    return (start, end)


def iterative_denoise(model, tokenizer, prompt, steps, top_p, temperature, simplex_value=5.0, max_length=128, tail_only=True):
    # Tokenize with offsets so we can build a mask that only covers the tail (3rd TAB field)
    enc = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    offsets = enc["offset_mapping"][0]  # (seq_len, 2)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Build span mask
    if tail_only:
        tail_span = _compute_tail_char_span(prompt)
        span_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if tail_span is not None:
            t_start, t_end = tail_span
            # mark token positions whose char offsets overlap with [t_start, t_end)
            for idx, (s, e) in enumerate(offsets.tolist()):
                if e > s and (t_start < e and t_end > s):
                    span_mask[0, idx] = True
        else:
            # Fallback: mask entire sequence if tail span cannot be resolved
            span_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        span_mask = torch.ones_like(input_ids, dtype=torch.bool)

    # Initial simplex representation from token ids (acts like noisy start).
    simplex = convert_to_simplex(input_ids, simplex_value, model.config.vocab_size)
    previous_pred = None

    logits = None
    for step in range(steps):
        # Random timestep (scalar) -> shape (batch,) float tensor
        timesteps = torch.randint(low=0, high=1000, size=(input_ids.shape[0],), device=device).float()
        with torch.no_grad():
            out = model(
                timesteps=timesteps,
                input_ids=input_ids,
                simplex=simplex,
                span_mask=span_mask,
                previous_pred=previous_pred,
                attention_mask=attention_mask,
                return_dict=True,
            )
        logits = out.logits  # (batch, seq_len, vocab)
        # Sample new simplex tokens (per position) using nucleus sampling.
        sampled_token_ids = sample_logits("top_p", logits, top_p=top_p, temperature=temperature)
        previous_pred = simplex
        simplex = convert_to_simplex(sampled_token_ids, simplex_value, model.config.vocab_size)

    # Final decode: take argmax of last logits for readability.
    if logits is None:
        return ""
    final_token_ids = torch.argmax(logits, dim=-1)
    text = tokenizer.batch_decode(final_token_ids, skip_special_tokens=True)
    return text[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default=[
            "Barack_Obama\tplace_of_birth\t<mask>\t1961-08-04",
            "Apple_Inc.\tfounded_by\t<mask>\t1976-04-01",
            "Cristiano_Ronaldo\tplays_for\t<mask>\t2021-08-27",
            "Amazon_(company)\theadquarters_location\t<mask>\t2020-12-31",
            "Angela_Merkel\tmember_of_political_party\t<mask>\t2005-11-22",
        ],
    )
    parser.add_argument("--prompts_file", type=str, default=None, help="Path to a file with one TAB-separated prompt per line (Head\tRelation\tTail\tDate)")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--tail_only", action="store_true", help="Mask only the tail (3rd TAB field)")
    parser.add_argument("--mask_all", dest="tail_only", action="store_false", help="Mask entire sequence instead of only tail")
    parser.set_defaults(tail_only=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, tokenizer = load_model_and_tokenizer(args.ckpt_dir, device)

    total, trainable = count_parameters(model)
    print(f"Model params total={total:,} trainable={trainable:,}")
    print(f"Config: hidden_size={model.config.hidden_size}, layers={model.config.num_hidden_layers}, heads={model.config.num_attention_heads}, vocab_size={model.config.vocab_size}")

    # Load prompts from file if provided
    prompts = []
    if args.prompts_file:
        if os.path.exists(args.prompts_file):
            with open(args.prompts_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prompts.append(line)
        else:
            print(f"[WARN] prompts_file not found: {args.prompts_file}; falling back to --prompts list")
    if not prompts:
        prompts = args.prompts

    for i, prompt in enumerate(prompts):
        print(f"\n=== Prompt {i+1}: {prompt}")
        gen = iterative_denoise(
            model,
            tokenizer,
            prompt,
            args.steps,
            args.top_p,
            args.temperature,
            max_length=args.max_length,
            tail_only=args.tail_only,
        )
        print(f"Generated (demo): {gen}")


if __name__ == "__main__":
    main()
