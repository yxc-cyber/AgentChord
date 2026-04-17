import threading
import time

import matplotlib.pyplot as plt
import torch
from transformers import BitsAndBytesConfig

from agentchord import GBC
from agentchord.gbc_object import GBCBase
from agentchord.model import ModelConfig, ModelFactory
from agentchord.utils import INPUT_FOOTER, INPUT_HEADER, INPUT_SEPARATOR

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

config = ModelConfig(
    local_model="Gemma3Model",
    model_path="/work/hdd/bghs/xyang7/models/gemma-3-27b-it",
    quantization_config=bnb_config,
    max_new_tokens=1024,
    do_sample=False,
    gradient_strategy="product_probs",
    connection_strategy="max_l1_norm",
    chat_template_path="src/agentchord/model/chat_templates/tool_chat_template_gemma3_json.jinja",
    enable_thinking=False,
)

model_factory = ModelFactory(config)
model = model_factory.create_model()


def sample_vram(samples, stop_event, interval_seconds: float = 0.1) -> None:
    if not torch.cuda.is_available():
        return

    while not stop_event.is_set():
        total_allocated = 0
        total_reserved = 0
        gpu_allocated = []
        gpu_reserved = []
        for device_idx in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
            resv = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
            total_allocated += alloc
            total_reserved += resv
            gpu_allocated.append(alloc)
            gpu_reserved.append(resv)
        samples.append((time.perf_counter(), total_allocated, total_reserved, gpu_allocated, gpu_reserved))
        time.sleep(interval_seconds)


def report_vram(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"{label}: CUDA is not available, VRAM usage cannot be reported.")
        return

    total_allocated = 0
    total_reserved = 0
    for device_idx in range(torch.cuda.device_count()):
        total_allocated += torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
        total_reserved += torch.cuda.memory_reserved(device_idx) / (1024 ** 2)

    num_devices = torch.cuda.device_count()
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(
        f"{label} (current device only): allocated={allocated:.2f} MiB, reserved={reserved:.2f} MiB, "
        f"peak_allocated={peak_allocated:.2f} MiB, peak_reserved={peak_reserved:.2f} MiB"
    )
    print(
        f"{label} (total across {num_devices} GPU(s)): allocated={total_allocated:.2f} MiB, reserved={total_reserved:.2f} MiB"
    )


input_content = GBC(
    value=f"{INPUT_HEADER}Input 1: A football match will be held tomorrow.{INPUT_SEPARATOR}Input 2: Weather Condition: Isolated thunderstorms throughout the day.{INPUT_SEPARATOR}Input 3: A cat sat on a mat.{INPUT_FOOTER}",
    connections=[
        "Input 1: A football match will be held tomorrow.",
        "Input 2: Weather Condition: Isolated thunderstorms throughout the day.",
        "Input 3: A cat sat on a mat.",
    ],
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful agent that can extract information about today's weather based on the input.",
    },
    {"role": "user", "content": input_content},
]

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

report_vram("Before completion")

print("=" * 80)
print("Running completion with dummy_weights=True")
print("=" * 80)
vram_samples_dummy = []
stop_event = threading.Event()
vram_thread = threading.Thread(target=sample_vram, args=(vram_samples_dummy, stop_event), daemon=True)

start_time_dummy = time.perf_counter()
vram_thread.start()
response_dummy = model.completion(messages, dummy_weights=True)
stop_event.set()
vram_thread.join()
end_time_dummy = time.perf_counter()
report_vram("After dummy_weights=True completion")
print(response_dummy)

output_dummy = response_dummy.choices[0].message.gbc_content
print(f"Is the output a GBC object? {isinstance(output_dummy, GBCBase)}")
print(f"Output weights: {output_dummy.get_weights()}")
print(f"All weights are 1.0? {all(weight == 1.0 for weight in output_dummy.get_weights())}")

print("=" * 80)
print("Running completion with dummy_weights=False")
print("=" * 80)
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

report_vram("Before completion")
vram_samples_normal = []
stop_event = threading.Event()
vram_thread = threading.Thread(target=sample_vram, args=(vram_samples_normal, stop_event), daemon=True)

start_time_normal = time.perf_counter()
vram_thread.start()
response_normal = model.completion(messages, dummy_weights=False)
stop_event.set()
vram_thread.join()
end_time_normal = time.perf_counter()
report_vram("After dummy_weights=False completion")
print(response_normal)

output_normal = response_normal.choices[0].message.gbc_content
print(f"Is the output a GBC object? {isinstance(output_normal, GBCBase)}")
print(f"Output weights: {output_normal.get_weights()}")

if torch.cuda.is_available() and (vram_samples_dummy or vram_samples_normal):
    num_gpus = torch.cuda.device_count()
    num_subplots = 1 + num_gpus
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 3 * num_subplots))
    if num_subplots == 1:
        axes = [axes]

    if vram_samples_dummy:
        times_dummy = [sample[0] - start_time_dummy for sample in vram_samples_dummy]
        allocated_dummy = [sample[1] for sample in vram_samples_dummy]
        reserved_dummy = [sample[2] for sample in vram_samples_dummy]
        axes[0].plot(times_dummy, allocated_dummy, label="dummy_weights=True (Allocated)", linewidth=2, color="blue")
        axes[0].plot(times_dummy, reserved_dummy, label="dummy_weights=True (Reserved)", linewidth=2, linestyle="--", color="blue")
        axes[0].axvline(end_time_dummy - start_time_dummy, color="blue", linestyle=":", linewidth=1, alpha=0.5)

    if vram_samples_normal:
        times_normal = [sample[0] - start_time_normal for sample in vram_samples_normal]
        allocated_normal = [sample[1] for sample in vram_samples_normal]
        reserved_normal = [sample[2] for sample in vram_samples_normal]
        axes[0].plot(times_normal, allocated_normal, label="dummy_weights=False (Allocated)", linewidth=2, color="red")
        axes[0].plot(times_normal, reserved_normal, label="dummy_weights=False (Reserved)", linewidth=2, linestyle="--", color="red")
        axes[0].axvline(end_time_normal - start_time_normal, color="red", linestyle=":", linewidth=1, alpha=0.5)

    axes[0].set_title("Total VRAM across all GPUs")
    axes[0].set_ylabel("VRAM (MiB)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    for gpu_idx in range(num_gpus):
        if vram_samples_dummy:
            times_dummy = [sample[0] - start_time_dummy for sample in vram_samples_dummy]
            gpu_allocated_dummy = [sample[3][gpu_idx] for sample in vram_samples_dummy]
            gpu_reserved_dummy = [sample[4][gpu_idx] for sample in vram_samples_dummy]
            axes[gpu_idx + 1].plot(times_dummy, gpu_allocated_dummy, label="dummy_weights=True (Allocated)", linewidth=2, color="blue")
            axes[gpu_idx + 1].plot(times_dummy, gpu_reserved_dummy, label="dummy_weights=True (Reserved)", linewidth=2, linestyle="--", color="blue")
            axes[gpu_idx + 1].axvline(end_time_dummy - start_time_dummy, color="blue", linestyle=":", linewidth=1, alpha=0.5)

        if vram_samples_normal:
            times_normal = [sample[0] - start_time_normal for sample in vram_samples_normal]
            gpu_allocated_normal = [sample[3][gpu_idx] for sample in vram_samples_normal]
            gpu_reserved_normal = [sample[4][gpu_idx] for sample in vram_samples_normal]
            axes[gpu_idx + 1].plot(times_normal, gpu_allocated_normal, label="dummy_weights=False (Allocated)", linewidth=2, color="red")
            axes[gpu_idx + 1].plot(times_normal, gpu_reserved_normal, label="dummy_weights=False (Reserved)", linewidth=2, linestyle="--", color="red")
            axes[gpu_idx + 1].axvline(end_time_normal - start_time_normal, color="red", linestyle=":", linewidth=1, alpha=0.5)

        gpu_name = torch.cuda.get_device_name(gpu_idx)
        axes[gpu_idx + 1].set_title(f"GPU {gpu_idx}: {gpu_name}")
        axes[gpu_idx + 1].set_ylabel("VRAM (MiB)")
        axes[gpu_idx + 1].legend(loc="upper left")
        axes[gpu_idx + 1].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time since completion start (s)")
    fig.suptitle("VRAM usage comparison: dummy_weights=True vs False", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("examples/gemma3_model_examples/gemma3_dummy_weights_vram_curve.png", dpi=200)
    plt.close()
    print("Saved VRAM comparison plot to examples/gemma3_model_examples/gemma3_dummy_weights_vram_curve.png")
elif not torch.cuda.is_available():
    print("VRAM curve was not generated because CUDA is not available.")

if torch.cuda.is_available():
    print(f"Total GPU count: {torch.cuda.device_count()}")
    for device_idx in range(torch.cuda.device_count()):
        print(f"  GPU {device_idx}: {torch.cuda.get_device_name(device_idx)}")
