"""Scan max context length for a regularly-loaded Llama model.

This script tries two scenarios:
 1) Inference-only (no gradients): uses model.generate with max_new_tokens=0 under torch.no_grad().
 2) Gradient-enabled (all inputs require grad): only attempted when the model's parameters all
    live on a single device (e.g., single GPU). It performs a forward with inputs_embeds that
    require grad and calls torch.autograd.grad on a simple scalar to allocate gradient memory.

Usage example:
  python3 scripts/context_length_scan_regular.py --model-path /path/to/model --device cuda:0 \
      --start 128 --step 128 --max 8192 --batch-size 1

Notes:
- For large models you may want to set --device to a single GPU (e.g., cuda:0) to allow the
  gradient-enabled scan to run. If the model is sharded across devices (device_map="auto"),
  the gradient-enabled scan will be skipped because cross-device autograd is unsupported here.
- The script attempts to be conservative and empties CUDA cache after failures.
"""

import argparse
import time
from importlib import util
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig


def load_model_regular(model_path, device_arg=None, quant_config: BitsAndBytesConfig = None):
    # Load model; prefer device_map='auto' and pass quantization_config when provided
    from transformers import AutoModelForCausalLM
    load_kwargs = {}
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    print("Loading model with device_map='auto' (transformers will place weights across devices)")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', **load_kwargs)

    # If a single device was requested, attempt to move whole model to that device
    if device_arg is not None and device_arg != "auto":
        try:
            device = torch.device(device_arg)
            print(f"Attempting to move model to device {device}")
            model.to(device)
        except Exception as e:
            print(f"Warning: failed to move model to {device_arg}: {e}")

    model.eval()
    return model


def try_generate_no_grad(model, input_ids):
    try:
        with torch.no_grad():
            out = model.generate(inputs=input_ids, max_new_tokens=1, do_sample=False, num_beams=1)
        return True, None
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return False, e
        raise


def try_forward_with_grad(model, input_ids, device):
    # Only attempt if all params are on the same device
    param_devices = {p.device for p in model.parameters()}
    if len(param_devices) != 1:
        return None, RuntimeError("Model parameters are on multiple devices; skipping grad-enabled test")
    model_device = next(iter(param_devices))
    input_ids = input_ids.to(model_device)
    # Build inputs_embeds and set requires_grad
    try:
        embed_layer = None
        # try to find the embedding layer
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_layer = model.model.embed_tokens
        elif hasattr(model, 'embed_tokens'):
            embed_layer = model.embed_tokens
        else:
            raise RuntimeError('Cannot find embed_tokens layer on model')

        inputs_embeds = embed_layer(input_ids).detach()
        inputs_embeds.requires_grad_(True)
        outputs = model(inputs_embeds=inputs_embeds, return_dict=True)
        # create simple scalar to backprop from
        logits = outputs.logits
        scalar = logits.pow(2).sum()
        # compute gradients w.r.t. inputs_embeds
        torch.autograd.grad(scalar, inputs_embeds)
        return True, None
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return False, e
        raise


def scan_lengths(model, tokenizer, start, step, max_len, batch_size, scenario):
    print(f"Starting scan for scenario: {scenario}")
    last_ok = None
    length = start
    while length <= max_len:
        print(f"Testing length={length}")
        pad_id = tokenizer.eos_token_id or 0
        input_ids = torch.full((batch_size, length), pad_id, dtype=torch.long)
        try:
            if scenario == 'no_grad':
                ok, err = try_generate_no_grad(model, input_ids)
            elif scenario == 'all_grad':
                ok, err = try_forward_with_grad(model, input_ids, None)
            else:
                raise ValueError('unknown scenario')
        except Exception as e:
            print(f"Error testing length={length}: {e}")
            break

        if ok is True:
            last_ok = length
            length += step
            continue
        elif ok is False:
            # OOM hit
            print(f"OOM at length={length}")
            break
        else:
            # ok is None meaning skipped (e.g., multi-device)
            print(f"Skipped scenario {scenario} (reason: {err})")
            last_ok = None
            break

    print(f"Finished scan for {scenario}, last successful length: {last_ok}")
    return last_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--device', default='auto', help="Device to load model on (e.g., 'cuda:0' or 'auto')")
    parser.add_argument('--start', type=int, default=128)
    parser.add_argument('--step', type=int, default=128)
    parser.add_argument('--max', type=int, default=8192)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_model_regular(args.model_path, device_arg=args.device, quant_config=BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16
    ))

    res_no_grad = scan_lengths(model, tokenizer, args.start, args.step, args.max, args.batch_size, 'no_grad')
    res_all_grad = scan_lengths(model, tokenizer, args.start, args.step, args.max, args.batch_size, 'all_grad')

    print('Results:')
    print('no_grad:', res_no_grad)
    print('all_grad:', res_all_grad)

if __name__ == '__main__':
    main()
