import argparse
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
        description="Compare full-sequence vs tail-window gradient memory usage for a causal LM."
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
        default="bnb4",
        help="Weight precision / quantization",
    )
    parser.add_argument(
        "--seq_lens",
        type=str,
        default="2048,4096,6144",
        help="Comma-separated list of total sequence lengths to test.",
    )
    parser.add_argument(
        "--tail_window",
        type=int,
        default=512,
        help="Tail window length (tokens) for gradient in tail-window mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (keep this at 1 for 70B…).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save VRAM in both modes.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Device map passed to from_pretrained (e.g. "auto").',
    )
    parser.add_argument(
        "--vocab_size_override",
        type=int,
        default=None,
        help="If set, use this fake vocab size for random ids instead of tokenizer.vocab_size.",
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
                bnb_4bit_compute_dtype=torch.float16,  # IMPORTANT: fp16 compute, not fp32
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

    # Disable cache by default; we'll override in prefill when needed
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

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


def run_full_seq_grad(
    model,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
):
    """
    Full-sequence gradients w.r.t. input embeddings (baseline).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

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

    inputs_embeds = inputs_embeds.detach().requires_grad_(True)

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # [B, S, V]
    last_logits = logits[:, -1, :]  # [B, V]
    loss = last_logits.max(dim=-1).values.sum()

    model.zero_grad(set_to_none=True)
    loss.backward()

    grads = inputs_embeds.grad
    assert grads is not None, "Full-seq: gradients for inputs_embeds are None!"

    return loss.item()


def run_tail_window_grad(
    model,
    vocab_size: int,
    total_seq_len: int,
    tail_window: int,
    batch_size: int,
):
    """
    Tail-window gradients:
    - Prefill long prefix under no_grad, using KV cache.
    - Backprop only through a short tail (tail_window tokens) with past_key_values.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    if tail_window > total_seq_len:
        raise ValueError(f"tail_window={tail_window} > total_seq_len={total_seq_len}")

    device0 = torch.device("cuda:0")
    prefix_len = total_seq_len - tail_window

    # Edge case: if prefix_len == 0, this is just full-seq on tail_window tokens.
    if prefix_len == 0:
        return run_full_seq_grad(model, vocab_size, tail_window, batch_size)

    # ----- 1) Prefill prefix under no_grad, building KV cache only -----
    prefix_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, prefix_len),
        device=device0,
        dtype=torch.long,
    )
    prefix_mask = torch.ones_like(prefix_ids, dtype=torch.long, device=device0)

    with torch.no_grad():
        prefill_out = model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            use_cache=True,              # explicitly build past_key_values
        )
        past_key_values = prefill_out.past_key_values

    # ----- 2) Tail with gradients w.r.t. tail embeddings only -----
    tail_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, tail_window),
        device=device0,
        dtype=torch.long,
    )

    embed = model.get_input_embeddings()
    with torch.no_grad():
        tail_embeds = embed(tail_ids)

    tail_embeds = tail_embeds.detach().requires_grad_(True)

    # Attention mask must have length prefix_len + tail_window
    full_len = prefix_len + tail_window
    attention_mask = torch.ones(
        (batch_size, full_len),
        device=device0,
        dtype=torch.long,
    )

    outputs = model(
        inputs_embeds=tail_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=False,              # no new cache needed for grad pass
    )
    logits = outputs.logits  # [B, tail_window, V]

    # Loss: max logit on the *last* tail token
    last_logits = logits[:, -1, :]  # [B, V]
    loss = last_logits.max(dim=-1).values.sum()

    model.zero_grad(set_to_none=True)
    loss.backward()

    grads = tail_embeds.grad
    assert grads is not None, "Tail-window: gradients for tail_embeds are None!"

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
            vocab_size = 32000
    print(f"Using vocab_size={vocab_size} for random input_ids")

    seq_lens: List[int] = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]
    print(f"Total sequence lengths to test: {seq_lens}")
    print(f"Batch size: {args.batch_size}, tail_window: {args.tail_window}")

    for seq_len in seq_lens:
        print("\n" + "=" * 80)
        print(f"Testing total_seq_len={seq_len}")

        # ----- Full-sequence baseline -----
        reset_memory_stats()
        try:
            loss_full = run_full_seq_grad(
                model=model,
                vocab_size=vocab_size,
                seq_len=seq_len,
                batch_size=args.batch_size,
            )
            print(f"[FULL] Loss (ignored): {loss_full:.4f}")
            print_memory(prefix=f"[FULL  total_len={seq_len}] AFTER forward+backward")
        except RuntimeError as e:
            print(f"[FULL  total_len={seq_len}] RuntimeError: {repr(e)}")
            print_memory(prefix=f"[FULL  total_len={seq_len}] AFTER OOM/ERROR")

        # ----- Tail-window mode -----
        reset_memory_stats()
        try:
            loss_tail = run_tail_window_grad(
                model=model,
                vocab_size=vocab_size,
                total_seq_len=seq_len,
                tail_window=args.tail_window,
                batch_size=args.batch_size,
            )
            print(f"[TAIL] Loss (ignored): {loss_tail:.4f}")
            print_memory(prefix=f"[TAIL  total_len={seq_len}, tail={args.tail_window}] AFTER forward+backward")
        except RuntimeError as e:
            print(f"[TAIL  total_len={seq_len}, tail={args.tail_window}] RuntimeError: {repr(e)}")
            print_memory(prefix=f"[TAIL  total_len={seq_len}, tail={args.tail_window}] AFTER OOM/ERROR")


if __name__ == "__main__":
    main()
