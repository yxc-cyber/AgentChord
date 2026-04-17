import threading
import time

import matplotlib.pyplot as plt
import torch
from transformers import BitsAndBytesConfig

from agentchord import GBC
from agentchord.gbc_object import GBCBase, visualize_gbc_tree
from agentchord.model import ModelConfig, ModelFactory
from agentchord.utils import INPUT_FOOTER, INPUT_HEADER, INPUT_SEPARATOR

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

config = ModelConfig(
    local_model="Qwen3Model",
    model_path="/shared/storage-01/users/xy61/models/Qwen3-32B",
    quantization_config=bnb_config,
    max_new_tokens=1024,
    do_sample=False,
    gradient_strategy="product_probs",
    connection_strategy="max_l1_norm",
    chat_template_path="src/agentchord/model/chat_templates/tool_chat_template_qwen3_json.jinja",
    enable_thinking=False,
)

model_factory = ModelFactory(config)
model = model_factory.create_model()


def sample_vram(samples, stop_event, interval_seconds: float = 0.1) -> None:
    if not torch.cuda.is_available():
        return

    while not stop_event.is_set():
        samples.append(
            (
                time.perf_counter(),
                torch.cuda.memory_allocated() / (1024 ** 2),
                torch.cuda.memory_reserved() / (1024 ** 2),
            )
        )
        time.sleep(interval_seconds)


def report_vram(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"{label}: CUDA is not available, VRAM usage cannot be reported.")
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(
        f"{label}: allocated={allocated:.2f} MiB, reserved={reserved:.2f} MiB, "
        f"peak_allocated={peak_allocated:.2f} MiB, peak_reserved={peak_reserved:.2f} MiB"
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

vram_samples = []
stop_event = threading.Event()
vram_thread = threading.Thread(target=sample_vram, args=(vram_samples, stop_event), daemon=True)

report_vram("Before completion")
start_time = time.perf_counter()
vram_thread.start()
response = model.completion(messages, dummy_weights=True)
stop_event.set()
vram_thread.join()
end_time = time.perf_counter()
report_vram("After completion")
print(response)

output = response.choices[0].message.gbc_content
print(f"Is the output a GBC object? {isinstance(output, GBCBase)}")
print(f"Class of the output: {output.__class__.__name__}")
print(f"Output weights: {output.get_weights()}")
print(f"All weights are 1.0? {all(weight == 1.0 for weight in output.get_weights())}")

if torch.cuda.is_available() and vram_samples:
    times = [sample[0] - start_time for sample in vram_samples]
    allocated = [sample[1] for sample in vram_samples]
    reserved = [sample[2] for sample in vram_samples]

    plt.figure(figsize=(10, 5))
    plt.plot(times, allocated, label="Allocated VRAM (MiB)", linewidth=2)
    plt.plot(times, reserved, label="Reserved VRAM (MiB)", linewidth=2)
    plt.axvline(0.0, color="gray", linestyle="--", linewidth=1, label="Start")
    plt.axvline(end_time - start_time, color="black", linestyle=":", linewidth=1, label="End")
    plt.title("VRAM usage during Qwen3 dummy-weights completion")
    plt.xlabel("Time since completion start (s)")
    plt.ylabel("VRAM (MiB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("examples/qwen3_model_examples/qwen3_dummy_weights_vram_curve.png", dpi=200)
    plt.close()
    print("Saved VRAM curve to examples/qwen3_model_examples/qwen3_dummy_weights_vram_curve.png")
elif not torch.cuda.is_available():
    print("VRAM curve was not generated because CUDA is not available.")

visualize_gbc_tree(
    response.choices[0].message.gbc_content,
    save_path="examples/qwen3_model_examples/qwen3_dummy_weights_example.png",
)