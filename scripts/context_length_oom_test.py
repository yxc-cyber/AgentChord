"""Find max context length before CUDA OOM for two scenarios:

1) No tokens require gradients (gradient_blocks = [[], ..., []])
2) All tokens require gradients (gradient_blocks = [[(0, L-1)], ..., [(0, L-1)]])

The script repeatedly calls the model's .generate() with increasing context length and
stops when a CUDA OOM (or RuntimeError caused by OOM) occurs. It prints the last
successful length for each scenario.

Usage:
    python3 scripts/context_length_oom_test.py \
        --model-path /path/to/your/model \
        --device cuda \
        --batch-size 1 \
        --start 128 --step 128 --max 8192

Notes:
- This script expects transformers + torch to be installed and a model compatible
  with the project's `LlamaForCausalLM_GBC` (the custom class in
  `src/agentchord/model/llama_model.py`).
- The script intentionally minimizes additional allocations: it calls
  `model.generate` with `max_new_tokens=0` (no new tokens) so it only performs a
  forward over the input context.
- If your model uses a custom device map (sharded across GPUs), `generate`
  handles device placement for input tensors.
"""

import argparse
import gc
import time

import torch
from transformers import BitsAndBytesConfig

from agentchord import GBC
from agentchord.gbc_object import GBCBase, visualize_gbc_tree
from agentchord.model import ModelConfig, ModelFactory
from agentchord.utils import INPUT_FOOTER, INPUT_HEADER, INPUT_SEPARATOR


def load_custom_llama_class(model_type, model_path, chat_template_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16
    )
    config = ModelConfig(
        local_model=model_type,
        model_path=model_path,
        quantization_config=bnb_config,
        max_new_tokens=1024,
        # temperature=0.0,
        do_sample=False,
        gradient_strategy="sum_squares",
        connection_strategy="mean_l1_norm",
        chat_template_path=chat_template_path
    )
    model_factory = ModelFactory(config)
    model = model_factory.create_model()

    return model, model.tokenizer


def try_length(model, tokenizer, batch_size, length, gradient_blocks, device, timeout_s=120):
    # build placeholder input_ids on CPU; generate will move as needed
    pad_id = tokenizer.eos_token_id if tokenizer is not None and tokenizer.eos_token_id is not None else 0
    input_ids = torch.full((batch_size, length), pad_id, dtype=torch.long)
    # call generate with minimal extra work
    try:
        start = time.time()
        # Keep generation config minimal: do not sample, no beams
        out = model.generate(
            inputs=input_ids,
            max_new_tokens=1,
            do_sample=False,
            num_beams=1,
            gradient_blocks=gradient_blocks,
        )
        dur = time.time() - start
        print(f"SUCCESS length={length} (time {dur:.2f}s)")
        return True, None
    except RuntimeError as e:
        # Detect CUDA OOM specifically
        msg = str(e)
        if "out of memory" in msg.lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            print(f"OOM at length={length}: {e.__class__.__name__}: {e}")
            # try to free cache
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return False, e
        else:
            # other runtime errors — propagate
            raise


def run_scan(model, tokenizer, batch_size, device, start, step, max_len, scenario_name, gradient_blocks_builder):
    print(f"Scanning scenario: {scenario_name}")
    last_success = None
    length = start
    while length <= max_len:
        blocks = gradient_blocks_builder(length, batch_size)
        ok, err = try_length(model, tokenizer, batch_size, length, blocks, device)
        if ok:
            last_success = length
            length += step
        else:
            # found OOM; stop scan
            break
    print(f"Scenario '{scenario_name}' last successful length: {last_success}")
    return last_success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="LlamaModel",
                        help="Local model class name to use (e.g. 'LlamaModel')")
    parser.add_argument("--model-path", default="/shared/storage-01/users/xy61/models/Llama-3.3-70B-Instruct",
                        help="Local model path or identifier")
    parser.add_argument("--chat-template-path", default="src/agentchord/model/chat_templates/tool_chat_template_llama3.2_json.jinja",
                        help="Path to the chat template Jinja file used by the model")
    parser.add_argument("--device", default="cuda", help="Device to use (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start", type=int, default=128)
    parser.add_argument("--step", type=int, default=128)
    parser.add_argument("--max", type=int, default=8192)
    parser.add_argument(
        "--setting",
        choices=["no_grad", "all_grad"],
        default="all_grad",
        help="Which scenario to run ('no_grad' or 'all_grad')."
    )
    args = parser.parse_args()

    # Scenario A: NO tokens require gradient -> pass empty blocks lists per batch element
    def no_grad_builder(length, batch_size):
        return [[] for _ in range(batch_size)]

    # Scenario B: ALL tokens require gradient -> pass blocks that cover the whole context per sample
    def all_grad_builder(length, batch_size):
        return [[(3*length//4, length)] for _ in range(batch_size)]

    device = args.device
    results = {}

    model, tokenizer = load_custom_llama_class(args.model_type, args.model_path, args.chat_template_path)
    model.model.eval()

    if args.setting == "no_grad":
        try:
            results['no_grad'] = run_scan(model.model, tokenizer, args.batch_size, device, args.start, args.step, args.max, "no_grad", no_grad_builder)
        except Exception as e:
            print(f"Error during no_grad scan: {e}")
    elif args.setting == "all_grad":
        # try:
        results['all_grad'] = run_scan(model.model, tokenizer, args.batch_size, device, args.start, args.step, args.max, "all_grad", all_grad_builder)
        # except Exception as e:
        #     print(f"Error during all_grad scan: {e}")
    else:
        print(f"Unknown setting: {args.setting}")

    print("Scan results:")
    print(results)


if __name__ == '__main__':
    main()
