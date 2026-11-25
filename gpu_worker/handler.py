"""RunPod Serverless handler for SDXL Turbo text-to-image and image-to-image."""

import base64
import binascii
import io
import os
from typing import Any, Dict, Optional

import runpod
import torch
from PIL import Image, UnidentifiedImageError
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/sdxl-turbo")
DEVICE = "cuda"
DTYPE = torch.float16
MIN_DIMENSION = 256
MAX_DIMENSION = 1024

text_pipe = None
image_pipe = None


def load_models() -> None:
    """Load model weights once and share components between both pipelines."""
    global text_pipe, image_pipe

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available; this worker requires a GPU")
    if text_pipe is not None and image_pipe is not None:
        return

    print(f"Loading {MODEL_ID} on {DEVICE}...")
    text_pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True,
    )
    text_pipe.to(DEVICE)
    image_pipe = AutoPipelineForImage2Image.from_pipe(text_pipe)
    image_pipe.to(DEVICE)

    # Memory optimizations supported by Diffusers.
    if hasattr(text_pipe, "enable_attention_slicing"):
        text_pipe.enable_attention_slicing()
    if hasattr(text_pipe, "enable_vae_slicing"):
        text_pipe.enable_vae_slicing()
    text_pipe.set_progress_bar_config(disable=True)
    image_pipe.set_progress_bar_config(disable=True)
    print("SDXL Turbo text-to-image and image-to-image pipelines are ready")


def _dimension(value: Any, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    number = max(MIN_DIMENSION, min(number, MAX_DIMENSION))
    return (number // 8) * 8


def _decode_reference(encoded: str, width: int, height: int) -> Image.Image:
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            return source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid reference_image_base64 image: {exc}") from exc


def _safe_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    return max(minimum, min(number, maximum))


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    data = (job or {}).get("input") or {}
    if not isinstance(data, dict):
        return {"success": False, "error": "input must be a JSON object"}

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return {"success": False, "error": "Prompt is required"}

    try:
        if text_pipe is None or image_pipe is None:
            load_models()

        width = _dimension(data.get("width", 512), "width")
        height = _dimension(data.get("height", 512), "height")
        steps = _safe_int(data.get("steps", 2), "steps", 1, 8)
        negative_prompt = str(data.get("negative_prompt", "")).strip() or None
        guidance = float(data.get("guidance_scale", 0.0))
        if not 0.0 <= guidance <= 10.0:
            raise ValueError("guidance_scale must be between 0 and 10")
        strength = float(data.get("strength", 0.62))
        if not 0.2 <= strength <= 0.9:
            raise ValueError("strength must be between 0.2 and 0.9")
        seed = int(data.get("seed", -1))
        generator: Optional[torch.Generator] = None
        if seed >= 0:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)

        reference = data.get("reference_image_base64")
        common = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": max(2, steps) if reference else steps,
            "guidance_scale": guidance,
            "generator": generator,
        }
        if reference:
            init_image = _decode_reference(str(reference), width, height)
            result = image_pipe(image=init_image, strength=strength, **common)
            mode = "image_to_image"
        else:
            result = text_pipe(width=width, height=height, **common)
            mode = "text_to_image"

        images = getattr(result, "images", None)
        if not images:
            raise RuntimeError("Diffusers returned no generated image")

        output = io.BytesIO()
        images[0].save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return {
            "success": True,
            "image_base64": encoded,
            "width": images[0].width,
            "height": images[0].height,
            "model": MODEL_ID,
            "mode": mode,
        }
    except Exception as exc:
        print(f"Generation failed: {type(exc).__name__}: {exc}")
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    load_models()
    runpod.serverless.start({"handler": handler})
