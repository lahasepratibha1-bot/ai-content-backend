"""Personal AI Video Studio GPU worker.

Adds a real text-to-video endpoint alongside the existing SDXL image endpoint.
Video backend: LTX-Video (legacy open-weight Diffusers pipeline).
This worker detects VRAM and refuses local video generation below the configured
minimum instead of silently returning a still-image slideshow.
"""
import base64
import gc
import os
import threading
import tempfile
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, LTXPipeline
from diffusers.utils import export_to_video

IMAGE_MODEL_ID = os.getenv("MODEL_ID", "stabilityai/sdxl-turbo")
VIDEO_MODEL_ID = os.getenv("VIDEO_MODEL_ID", "Lightricks/LTX-Video")
TOKEN = os.getenv("IMAGE_WORKER_TOKEN", "")
MIN_VIDEO_VRAM_GB = float(os.getenv("MIN_VIDEO_VRAM_GB", "10"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Personal AI Video Studio GPU Worker", version="4.0.0")
image_pipe = None
image_to_image_pipe = None
video_pipe = None
active_model = "none"
model_lock = threading.Lock()

class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=6000)
    negative_prompt: str = ""
    width: int = Field(default=576, ge=256, le=1024)
    height: int = Field(default=1024, ge=256, le=1536)
    steps: int = Field(default=4, ge=1, le=8)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=10.0)
    seed: int = -1
    reference_image_base64: Optional[str] = None
    strength: float = Field(default=0.62, ge=0.2, le=0.9)

class VideoRequest(BaseModel):
    prompt: str = Field(min_length=5, max_length=6000)
    negative_prompt: str = "low quality, blurry, jittery motion, deformed face, extra limbs, flicker, text, watermark"
    width: int = Field(default=512, ge=320, le=1024)
    height: int = Field(default=896, ge=320, le=1024)
    num_frames: int = Field(default=97, ge=49, le=145)
    fps: int = Field(default=24, ge=8, le=30)
    steps: int = Field(default=30, ge=8, le=50)
    guidance_scale: float = Field(default=3.0, ge=1.0, le=5.0)
    seed: int = -1


def auth(value):
    if TOKEN and value != "Bearer " + TOKEN:
        raise HTTPException(401, "Invalid worker token.")


def gpu_info():
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device": "cpu", "gpu_name": None, "vram_total_gb": 0.0,
                "vram_free_gb": 0.0, "video_generation_supported": False,
                "reason": "CUDA GPU is not available on this worker."}
    props = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    total_gb = round(total_bytes / (1024**3), 2)
    free_gb = round(free_bytes / (1024**3), 2)
    supported = total_gb >= MIN_VIDEO_VRAM_GB
    return {"cuda_available": True, "device": "cuda", "gpu_name": props.name,
            "vram_total_gb": total_gb, "vram_free_gb": free_gb,
            "video_generation_supported": supported,
            "video_model": VIDEO_MODEL_ID if supported else None,
            "reason": None if supported else f"Local LTX video generation needs at least {MIN_VIDEO_VRAM_GB} GB VRAM; this GPU reports {total_gb} GB."}


def clear_models():
    global image_pipe, image_to_image_pipe, video_pipe, active_model
    image_pipe = None
    image_to_image_pipe = None
    video_pipe = None
    active_model = "none"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_image_models():
    global image_pipe, image_to_image_pipe, active_model
    if active_model == "image" and image_pipe is not None:
        return
    clear_models()
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(IMAGE_MODEL_ID, torch_dtype=dtype, use_safetensors=True)
    pipe.to(DEVICE)
    image_pipe = pipe
    image_to_image_pipe = AutoPipelineForImage2Image.from_pipe(pipe)
    image_to_image_pipe.to(DEVICE)
    if DEVICE == "cuda":
        image_pipe.enable_attention_slicing()
        image_to_image_pipe.enable_attention_slicing()
    active_model = "image"


def load_video_model():
    global video_pipe, active_model
    if video_pipe is not None and active_model == "video":
        return
    info = gpu_info()
    if not info["video_generation_supported"]:
        raise RuntimeError(info["reason"])
    clear_models()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = LTXPipeline.from_pretrained(VIDEO_MODEL_ID, torch_dtype=dtype)
    # Offload is safer for unknown GPU models; generation may be slower than full-GPU mode.
    pipe.enable_model_cpu_offload()
    video_pipe = pipe
    active_model = "video"


@app.get("/health")
def health():
    info = gpu_info()
    return {"status": "healthy", "version": "4.0.0", "image_model": IMAGE_MODEL_ID,
            "video_model": VIDEO_MODEL_ID, "active_model": active_model,
            "models_loaded": {"image": image_pipe is not None, "video": video_pipe is not None}, **info}


@app.post("/generate")
def generate_image(data: ImageRequest, authorization: Optional[str] = Header(default=None)):
    global image_pipe, image_to_image_pipe
    auth(authorization)
    with model_lock:
        try:
            load_image_models()
            gen = torch.Generator(device=DEVICE).manual_seed(data.seed) if data.seed >= 0 else None
            with torch.inference_mode():
                if data.reference_image_base64:
                    from PIL import Image
                    import io
                    raw = base64.b64decode(data.reference_image_base64)
                    init = Image.open(io.BytesIO(raw)).convert("RGB").resize((data.width, data.height))
                    result = image_to_image_pipe(prompt=data.prompt, negative_prompt=data.negative_prompt or None,
                        image=init, strength=data.strength, num_inference_steps=max(2, data.steps),
                        guidance_scale=data.guidance_scale, generator=gen)
                else:
                    result = image_pipe(prompt=data.prompt, negative_prompt=data.negative_prompt or None,
                        width=data.width, height=data.height, num_inference_steps=data.steps,
                        guidance_scale=data.guidance_scale, generator=gen)
            import io
            out = io.BytesIO()
            result.images[0].save(out, format="PNG")
            return {"success": True, "image_base64": base64.b64encode(out.getvalue()).decode("utf-8"),
                    "width": result.images[0].width, "height": result.images[0].height, "model": IMAGE_MODEL_ID}
        except Exception as exc:
            raise HTTPException(500, f"Image generation error: {exc}")


@app.post("/generate-video")
def generate_video(data: VideoRequest, authorization: Optional[str] = Header(default=None)):
    auth(authorization)
    with model_lock:
        try:
            load_video_model()
            gen = torch.Generator(device="cuda").manual_seed(data.seed) if data.seed >= 0 else None
            with torch.inference_mode():
                result = video_pipe(prompt=data.prompt, negative_prompt=data.negative_prompt,
                    width=data.width, height=data.height, num_frames=data.num_frames,
                    num_inference_steps=data.steps, guidance_scale=data.guidance_scale,
                    generator=gen, output_type="pil")
            frames = result.frames[0]
            with tempfile.TemporaryDirectory() as tmp:
                mp4_path = Path(tmp) / "scene.mp4"
                export_to_video(frames, str(mp4_path), fps=data.fps)
                video_bytes = mp4_path.read_bytes()
            return {"success": True, "video_base64": base64.b64encode(video_bytes).decode("utf-8"),
                    "mime_type": "video/mp4", "frames": len(frames), "fps": data.fps,
                    "duration_seconds": round(len(frames) / data.fps, 2), "model": VIDEO_MODEL_ID}
        except Exception as exc:
            raise HTTPException(500, f"Video generation error: {exc}")
