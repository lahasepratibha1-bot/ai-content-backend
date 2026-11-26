"""Run a real SDXL Turbo inference test on a CUDA GPU.

Usage (inside the GPU worker environment):
    python gpu_smoke_test.py

The test uses the worker's own model loader and writes gpu_smoke_test.png.
It intentionally fails on CPU-only hosts; use the Docker smoke workflow for CPU CI.
"""
import gc
import sys
import time
from pathlib import Path

import torch
from PIL import Image

import main as worker


def main():
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA GPU is not available; this test requires a real GPU.")

    props = torch.cuda.get_device_properties(0)
    free_before, total = torch.cuda.mem_get_info(0)
    print(f"GPU: {props.name}")
    print(f"VRAM total: {total / (1024 ** 3):.2f} GB")
    print(f"VRAM free before load: {free_before / (1024 ** 3):.2f} GB")
    print(f"Image model: {worker.IMAGE_MODEL_ID}")

    started = time.time()
    worker.load_image_models()
    load_seconds = time.time() - started
    print(f"Model load completed in {load_seconds:.1f}s")

    generator = torch.Generator(device="cuda").manual_seed(20261126)
    started = time.time()
    with torch.inference_mode():
        result = worker.image_pipe(
            prompt=(
                "A warm cinematic portrait of a friendly young woman reading a book "
                "beside a sunlit window, realistic photography, natural expression"
            ),
            width=512,
            height=512,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
        )
    inference_seconds = time.time() - started

    image = result.images[0]
    if not isinstance(image, Image.Image):
        raise RuntimeError("Pipeline did not return a PIL image.")
    if image.size != (512, 512):
        raise RuntimeError(f"Unexpected image size: {image.size}")

    output = Path(__file__).resolve().parent / "gpu_smoke_test.png"
    image.save(output, format="PNG")
    if not output.exists() or output.stat().st_size < 1024:
        raise RuntimeError("Generated PNG is missing or unexpectedly small.")

    free_after, _ = torch.cuda.mem_get_info(0)
    print("PASS: SDXL Turbo generated a valid 512x512 PNG.")
    print(f"Model load time: {load_seconds:.1f}s")
    print(f"Inference time: {inference_seconds:.1f}s")
    print(f"VRAM free after inference: {free_after / (1024 ** 3):.2f} GB")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
