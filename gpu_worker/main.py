
import base64
import binascii
import io
import os
import torch

from typing import Optional
from PIL import Image
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/sdxl-turbo")
TOKEN = os.getenv("IMAGE_WORKER_TOKEN", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="ViralVerse AI Studio GPU Worker")

text_pipe = None
img_pipe = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=6000)
    negative_prompt: str = Field(default="", max_length=2000)
    width: int = Field(default=576, ge=256, le=1024)
    height: int = Field(default=1024, ge=256, le=1024)
    steps: int = Field(default=1, ge=1, le=8)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=10.0)
    seed: int = -1
    reference_image_base64: Optional[str] = None
    strength: float = Field(default=0.62, ge=0.2, le=0.9)


def auth(value):
    if not TOKEN:
        raise HTTPException(503, "Worker authentication is not configured.")

    if value != "Bearer " + TOKEN:
        raise HTTPException(401, "Invalid worker token.")


@app.on_event("startup")
def load_models():
    global text_pipe, img_pipe

    if not TOKEN:
        raise RuntimeError("IMAGE_WORKER_TOKEN environment variable is required.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. CPU fallback is disabled.")

    try:
        kwargs = {
            "torch_dtype": torch.float16,
            "use_safetensors": True
        }

        text_pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID, **kwargs
        ).to("cuda")

        # Reuse loaded components to reduce duplicate GPU memory usage.
        img_pipe = AutoPipelineForImage2Image.from_pipe(text_pipe)

        text_pipe.enable_attention_slicing()
        img_pipe.enable_attention_slicing()

        print("GPU worker model loaded:", MODEL_ID)
        print("GPU:", torch.cuda.get_device_name(0))

    except Exception as exc:
        text_pipe = None
        img_pipe = None
        raise RuntimeError("Model loading failed: " + str(exc))


@app.get("/health")
def health():
    return {
        "status": "healthy" if text_pipe is not None else "loading",
        "model": MODEL_ID,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None
        )
    }


@app.post("/generate")
def generate(
    data: GenerateRequest,
    authorization: Optional[str] = Header(default=None)
):
    auth(authorization)

    if text_pipe is None or img_pipe is None:
        raise HTTPException(503, "Models are not ready.")

    generator = None
    if data.seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(data.seed)

    try:
        with torch.inference_mode():

            if data.reference_image_base64:
                try:
                    raw = base64.b64decode(
                        data.reference_image_base64,
                        validate=True
                    )
                    init = Image.open(io.BytesIO(raw)).convert("RGB")
                    init = init.resize((data.width, data.height))
                except (ValueError, binascii.Error, OSError):
                    raise HTTPException(400, "Invalid reference image.")

                result = img_pipe(
                    prompt=data.prompt,
                    negative_prompt=data.negative_prompt or None,
                    image=init,
                    strength=data.strength,
                    num_inference_steps=max(2, data.steps),
                    guidance_scale=data.guidance_scale,
                    generator=generator
                )

            else:
                result = text_pipe(
                    prompt=data.prompt,
                    negative_prompt=data.negative_prompt or None,
                    width=data.width,
                    height=data.height,
                    num_inference_steps=data.steps,
                    guidance_scale=data.guidance_scale,
                    generator=generator
                )

            image = result.images[0]
            output = io.BytesIO()
            image.save(output, format="PNG")

            return {
                "success": True,
                "image_base64": base64.b64encode(
                    output.getvalue()
                ).decode("utf-8"),
                "width": image.width,
                "height": image.height,
                "model": MODEL_ID
            }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            500,
            "Image generation failed: " + str(exc)[:500]
        )
