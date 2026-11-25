import base64
import binascii
import io
import os
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/sdxl-turbo")
TOKEN = os.getenv("IMAGE_WORKER_TOKEN", "").strip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="Personal AI Video Studio GPU Worker", version="1.1.0")
text_pipe = None
img_pipe = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=6000)
    negative_prompt: str = Field(default="", max_length=2000)
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=768, ge=256, le=1024)
    steps: int = Field(default=2, ge=1, le=8)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=10.0)
    seed: int = -1
    reference_image_base64: Optional[str] = None
    strength: float = Field(default=0.62, ge=0.2, le=0.9)


def check_auth(value: Optional[str]) -> None:
    # The worker is exposed through a public tunnel; never allow unauthenticated calls.
    if not TOKEN:
        raise HTTPException(status_code=503, detail="IMAGE_WORKER_TOKEN is not configured.")
    if value != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid worker token.")


@app.on_event("startup")
def load_models() -> None:
    global text_pipe, img_pipe

    if not TOKEN:
        raise RuntimeError("Set IMAGE_WORKER_TOKEN before starting the public GPU worker.")

    load_kwargs = {"torch_dtype": DTYPE, "use_safetensors": True}
    try:
        # Load the model weights once. The image-to-image pipeline reuses the
        # text-to-image components instead of loading a second full SDXL model.
        text_pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, **load_kwargs)
        text_pipe.to(DEVICE)
        img_pipe = AutoPipelineForImage2Image.from_pipe(text_pipe)

        if DEVICE == "cuda":
            text_pipe.enable_attention_slicing()
            if hasattr(text_pipe, "enable_vae_slicing"):
                text_pipe.enable_vae_slicing()
        text_pipe.set_progress_bar_config(disable=True)
        img_pipe.set_progress_bar_config(disable=True)
    except Exception:
        text_pipe = None
        img_pipe = None
        raise


@app.get("/health")
def health():
    return {
        "status": "healthy" if text_pipe is not None and img_pipe is not None else "loading",
        "model": MODEL_ID,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "text_to_image_ready": text_pipe is not None,
        "image_to_image_ready": img_pipe is not None,
    }


@app.post("/generate")
def generate(data: GenerateRequest, authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)
    if text_pipe is None or img_pipe is None:
        raise HTTPException(status_code=503, detail="GPU models are not ready.")
    if data.width % 8 != 0 or data.height % 8 != 0:
        raise HTTPException(status_code=422, detail="Width and height must be multiples of 8.")

    generator = None
    if data.seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(data.seed)

    try:
        common = {
            "prompt": data.prompt,
            "negative_prompt": data.negative_prompt or None,
            "num_inference_steps": max(2, data.steps) if data.reference_image_base64 else data.steps,
            "guidance_scale": data.guidance_scale,
            "generator": generator,
        }

        if data.reference_image_base64:
            try:
                raw = base64.b64decode(data.reference_image_base64, validate=True)
                with Image.open(io.BytesIO(raw)) as source:
                    source.load()
                    init_image = source.convert("RGB").resize((data.width, data.height))
            except (binascii.Error, UnidentifiedImageError, OSError, ValueError):
                raise HTTPException(status_code=400, detail="Invalid reference_image_base64 image.")

            result = img_pipe(image=init_image, strength=data.strength, **common)
        else:
            result = text_pipe(width=data.width, height=data.height, **common)

        if not result.images:
            raise RuntimeError("Diffusers returned an empty image list.")

        output = io.BytesIO()
        result.images[0].save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return {
            "success": True,
            "image_base64": encoded,
            "width": result.images[0].width,
            "height": result.images[0].height,
            "model": MODEL_ID,
            "mode": "image_to_image" if data.reference_image_base64 else "text_to_image",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation error: {type(exc).__name__}: {exc}")
