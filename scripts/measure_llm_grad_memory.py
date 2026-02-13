import argparse
import math
import os
import sys
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure VRAM usage for LLM gradients w.r.t input embeddings."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/shared/storage-01/users/xy61/models/Llama-3.3-70B-Instruct",
        help="HF model id or local path",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16", "bnb4", "bnb8"],
        default="fp16",
        help="Weight precision / quantization",
    )
    parser.add_argument(
        "--seq_lens",
        type=str,
        default="1024,2048,4096,6144,8192",
        help="Comma-separated list of sequence lengths to test.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save VRAM.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (careful, memory grows linearly with this).",
    )
    parser.add_argument(
        "--vocab_size_override",
        type=int,
        default=None,
        help="If set, use this fake vocab size for random ids instead of tokenizer.vocab_size.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Device map passed to from_pretrained (e.g. "auto").',
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, precision: str, device_map: str, gradient_checkpointing: bool):
    kwargs = {"device_map": device_map}

    if precision in ("fp16", "bf16"):
        kwargs["torch_dtype"] = torch.float16 if precision == "fp16" else torch.bfloat16
    elif precision in ("bnb4", "bnb8"):
        if not HAS_BNB:
            raise RuntimeError("bitsandbytes not available, but precision is set to bnb4/bnb8.")
        if precision == "bnb4":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,  # use fp16 compute
            )
        else:
            quant_cfg = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        kwargs["quantization_config"] = quant_cfg
        kwargs["torch_dtype"] = torch.float16
    else:
        raise ValueError(f"Unknown precision: {precision}")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    print(f"Loading model: {model_name} with precision={precision}, device_map={device_map}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    # Disable cache for proper gradient flow and to avoid extra KV memory
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Optionally enable grad checkpointing
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    return model, tokenizer


def reset_memory_stats():
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def print_memory(prefix: str):
    if not torch.cuda.is_available():
        print(f"{prefix}: CUDA not available, skipping memory stats.")
        return
    n_devices = torch.cuda.device_count()
    for i in range(n_devices):
        alloc = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)
        print(f"{prefix} | GPU {i}: allocated_peak={alloc:.2f} GB, reserved_peak={reserved:.2f} GB")


def run_one_experiment(
    model,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
):
    """
    Full-sequence gradients w.r.t. input embeddings.

    Steps:
    - create random input_ids
    - embed -> inputs_embeds (requires_grad=True for full sequence)
    - forward with use_cache=False
    - loss = last-token logit sum
    - backward()
    """
    if not torch.cuda.is_available():
        raise RuntimeError("This script assumes CUDA is available.")

    # Simple random ids; we don't care about real text for VRAM measurement
    device0 = torch.device("cuda:0")
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device0,
        dtype=torch.long,
    )

    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device0)

    embed = model.get_input_embeddings()
    with torch.no_grad():
        inputs_embeds = embed(input_ids)

    # We want gradients w.r.t. ALL tokens (full sequence)
    inputs_embeds = inputs_embeds.detach().requires_grad_(True)

    # Forward; important: use inputs_embeds and use_cache=False
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits  # [B, S, V]
    # Take scalar loss from last token logits; just pick the max-logit token position
    last_logits = logits[:, -1, :]  # [B, V]
    loss = last_logits.max(dim=-1).values.sum()

    # Zero grads to avoid accumulation
    model.zero_grad(set_to_none=True)

    # Backward: this will populate inputs_embeds.grad, not model parameter grads
    loss.backward()

    # Just to be sure grads are there (and to prevent any lazy behavior)
    grads = inputs_embeds.grad
    assert grads is not None, "Gradients for inputs_embeds are None!"

    return loss.item()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this script is meant to run on GPUs.")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        precision=args.precision,
        device_map=args.device_map,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Determine vocab size for random ids
    if args.vocab_size_override is not None:
        vocab_size = args.vocab_size_override
    else:
        if hasattr(tokenizer, "vocab_size"):
            vocab_size = tokenizer.vocab_size
        elif hasattr(model.config, "vocab_size"):
            vocab_size = model.config.vocab_size
        else:
            vocab_size = 32000  # fallback
    print(f"Using vocab_size={vocab_size} for random input_ids")

    seq_lens: List[int] = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]
    print(f"Sequence lengths to test: {seq_lens}")
    print(f"Batch size: {args.batch_size}")

    for seq_len in seq_lens:
        print("\n" + "=" * 80)
        print(f"Testing seq_len={seq_len}")
        reset_memory_stats()

        try:
            loss_val = run_one_experiment(
                model=model,
                vocab_size=vocab_size,
                seq_len=seq_len,
                batch_size=args.batch_size,
            )
            print(f"Loss (ignored, just for sanity): {loss_val:.4f}")
            print_memory(prefix=f"[seq_len={seq_len}] AFTER forward+backward")
        except RuntimeError as e:
            print(f"[seq_len={seq_len}] RuntimeError: {repr(e)}")
            print_memory(prefix=f"[seq_len={seq_len}] AFTER OOM/ERROR")
            # If it's an OOM, it's probably going to fail for larger seq_lens too
            if "out of memory" in str(e).lower():
                print("Encountered CUDA OOM; stopping sweep.")
                break


if __name__ == "__main__":
    main()
