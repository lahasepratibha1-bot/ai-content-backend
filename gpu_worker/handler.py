"""RunPod serverless handler for SDXL Turbo text-to-image and image-to-image."""

import base64
import binascii
import io
import os
from typing import Any, Dict

import runpod
import torch
from PIL import Image, UnidentifiedImageError
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/sdxl-turbo")
DEVICE = "cuda"
DTYPE = torch.float16
MAX_DIMENSION = 1024
MIN_DIMENSION = 256

text_pipe = None
image_pipe = None


def load_models() -> None:
    """Load SDXL once and create the img2img pipeline from shared components."""
    global text_pipe, image_pipe

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available; RunPod worker requires a GPU")

    if text_pipe is not None and image_pipe is not None:
        return

    print(f"Loading {MODEL_ID} on {DEVICE}...")
    text_pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True,
    )
    text_pipe.to(DEVICE)

    # Reuse the text pipeline's components rather than loading a second model.
    image_pipe = AutoPipelineForImage2Image.from_pipe(text_pipe)
    image_pipe.to(DEVICE)

    # Reduce peak memory use where supported by the installed Diffusers version.
    if hasattr(text_pipe, "enable_attention_slicing"):
        text_pipe.enable_attention_slicing()
    if hasattr(text_pipe, "enable_vae_slicing"):
        text_pipe.enable_vae_slicing()

    print("SDXL Turbo text-to-image and image-to-image pipelines are ready")


def _bounded_dimension(value: Any, field_name: str) -> int:
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer")

    dimension = max(MIN_DIMENSION, min(dimension, MAX_DIMENSION))
    # SDXL dimensions must be divisible by 8.
    return max(8, (dimension // 8) * 8)


def _decode_reference_image(encoded: str, width: int, height: int) -> Image.Image:
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            return source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"reference_image_base64 is not a valid image: {exc}") from exc


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    data = job.get("input") or {}
    if not isinstance(data, dict):
        return {"success": False, "error": "input must be a JSON object"}

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return {"success": False, "error": "Prompt is required"}

    try:
        if text_pipe is None or image_pipe is None:
            load_models()

        negative_prompt = str(data.get("negative_prompt", "")).strip() or None
        width = _bounded_dimension(data.get("width", 512), "width")
        height = _bounded_dimension(data.get("height", 512), "height")
        steps = max(1, min(int(data.get("steps", 2)), 8))
        guidance = float(data.get("guidance_scale", 0.0))
        strength = max(0.2, min(float(data.get("strength", 0.62)), 0.9))
        seed = int(data.get("seed", -1))

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)

        reference = data.get("reference_image_base64")
        if reference:
            init_image = _decode_reference_image(str(reference), width, height)
            result = image_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=max(2, steps),
                guidance_scale=guidance,
                generator=generator,
            )
        else:
            result = text_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

        if not getattr(result, "images", None):
            raise RuntimeError("Diffusers returned no generated image")

        output = io.BytesIO()
        result.images[0].save(output, format="PNG")
